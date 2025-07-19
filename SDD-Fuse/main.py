
import os
import warnings
import numpy as np
import wandb
import logging
import torch
import FR_Net
from torch.nn import init
from torchvision.utils import make_grid, save_image
from spikingjelly.activation_based import functional
from tqdm import trange
import random
import torch.optim as optim
from Scheduler import GradualWarmupScheduler
logger = logging.getLogger('base')

from diffusion import GaussianDiffusionTrainer,GaussianDiffusionSampler
from model import Spk_UNet
import data as Data
import core.metrics as Metrics

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--train', action='store_true', default=False, help='train from scratch')
parser.add_argument('--eval', action='store_true', default=False, help='load Spk_UNet.pt and FR_net.pt')
parser.add_argument('--dataset', type=str, default='', choices=['MSRS', 'Harvard', 'Roadscene'],help='dataset name')
parser.add_argument('--sample_type', type=str, default='', choices=['ddpm', 'ddim', 'ddpm2'], help='Sample Type')
parser.add_argument('--wandb', action='store_true', default=False, help='use wandb to log training')
# Spiking UNet
parser.add_argument('--ch', default='', type=int, choices=[64, 128], help='base channel of UNet')
parser.add_argument('--ch_mult', default=[1, 2, 3, 4], help='channel multiplier')
parser.add_argument('--attn', default=[], help='No attention mechanism')
parser.add_argument('--num_res_blocks', default=2, type=int, help='# resblock in each level')
parser.add_argument('--img_size', default=128, type=int, help='image size')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of resblock')
parser.add_argument('--timestep', default='', type=int, choices=[4, 6, 8], help='snn timestep')
parser.add_argument('--img_ch', type=int, default=3, help='image channel')
# Gaussian Diffusion
parser.add_argument('--beta_1', default=1e-4, type=float, help='start beta value')
parser.add_argument('--beta_T', default=0.02, type=float, help='end beta value')
parser.add_argument('--T', default=1000, type=int, help='total diffusion steps')
parser.add_argument('--mean_type', default='epsilon', help='predict variable:[xprev, xstart, epsilon]')
parser.add_argument('--var_type', default='fixedlarge', help='variance type:[fixedlarge, fixedsmall]')
# Training
parser.add_argument('--resume', default=False, help="load pre-trained model")
parser.add_argument('--resume_model', type=str, default='', help='resume model path')
parser.add_argument('--lr', default=2e-4, help='target learning rate')
parser.add_argument('--grad_clip', default=1., help="gradient norm clipping")
parser.add_argument('--total_steps', type=int, default=20001, help='total training steps')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='workers of Dataloader')
parser.add_argument('--parallel', default=True, help='multi gpu training')
# Logging & Sampling
parser.add_argument('--logdir', default='./logs', help='log directory')
parser.add_argument('--sample_size', type=int,default=4, help="sampling size of images")
parser.add_argument('--sample_step', type=int,default=5000, help='frequency of sampling')
# Evaluation
parser.add_argument('--save_step', type=int,default=2000, help='frequency of saving checkpoints, 0 to disable during training')
parser.add_argument('--pre_trained_path', default="", help='Fusion Performance')
parser.add_argument('--multiplier', type=float, default=2.)

args = parser.parse_args()


device = torch.device('cuda')

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = False


def infiniteloop(dataloader):
    while True:
        for batch in iter(dataloader):
            yield batch


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('TSMConv2d') != -1:
        init.orthogonal_(m.tsmconv.weight.data, gain=1)
        if m.tsmconv.bias is not None:
            m.tsmconv.bias.data.zero_()
    elif classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


def train():

    if args.dataset == 'MSRS':
        dataset = Data.create_dataset(root='home/dataset/MSRS',dataname='MSRS')
        
    elif args.dataset == 'Harvard':
        dataset = Data.create_dataset(root='home/dataset/Harvard',dataname='Harvard')

    else:
        raise NotImplementedError

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True, timeout=60)
                                        
    datalooper = infiniteloop(dataloader)

    print(f'-------Starting loading {args.dataset} Dataset!-------')
    

    # model setup
    net_model = Spk_UNet(
       T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
       num_res_blocks=args.num_res_blocks, dropout=args.dropout, timestep=args.timestep, img_ch=args.img_ch)

    refinement_fn = FR_Net.Restormer_ROPE(in_channel=1, out_channel=1)
    init_weights(refinement_fn, init_type='orthogonal')
    optimizer = optim.AdamW(list(net_model.parameters()) + list(refinement_fn.parameters()), lr=args.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.total_steps, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=args.multiplier, warm_epoch=args.total_steps // 10, after_scheduler=cosineScheduler)
    if args.resume:
        ckpt = torch.load(os.path.join(args.resume_model))
        print(f'Loading Resume model from {args.resume_model}')
        net_model.load_state_dict(ckpt['net_model'], strict=True)
        refinement_fn.load_state_dict(ckpt['refinement_fn'], strict=True)
    else:
        print('Training from scratch')
        

    trainer = GaussianDiffusionTrainer(
    net_model, refinement_fn, float(args.beta_1), float(args.beta_T), args.T).to(device)

    net_sampler = GaussianDiffusionSampler(
        net_model, refinement_fn, float(args.beta_1), float(args.beta_T), args.T, args.img_size,
        args.mean_type, args.var_type).to(device)
    
    if args.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler).cuda()




    # log setup
    if not os.path.exists(os.path.join(args.logdir,'sample')):
        os.makedirs(os.path.join(args.logdir, 'sample'))
    x_T = torch.randn(int(args.sample_size), int(args.img_ch), int(args.img_size), int(args.img_size))
    x_T = x_T.to(device)
    # grid = (make_grid(next(iter(dataloader))[0][:args.sample_size]) + 1) / 2
    # save_image(grid, os.path.join(args.logdir,'sample','groundtruth.png'))

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    
    # start training
    with trange(args.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar :
            # train
            optimizer.zero_grad()
            x_0 = next(datalooper)
            loss = trainer(x_0).mean()
            loss.backward()

            if args.wandb:
                wandb.log({'training loss': loss.item()})

            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), args.grad_clip)
            optimizer.step()
            warmUpScheduler.step()
            pbar.set_postfix(loss='%.5f' % loss)

            ## reset SNN neuron
            functional.reset_net(net_model)

            # sample
            print(f'Sample at {step} step')
            if args.sample_step > 0 and step % args.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = net_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        args.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    ## log to wandb
                    if args.wandb:
                        wandb.log({'sample': [wandb.Image(grid, caption='sample')]})

                net_model.train()

            # save
            # print(f'Save model at {step} step')
            if args.save_step > 0 and step % args.save_step == 0 and step > 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'refinement_fn':refinement_fn.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                save_path = str(step) +  'ckpt.pt'
                torch.save(ckpt, os.path.join(args.logdir,save_path))
    


def eval():
    # model setup

    dataset = Data.create_dataset(root='home/dataset/Harvard', dataname='Test_mif')
    # dataset = Data.create_dataset(root='home/dataset/MSRS', dataname='Test_vif')
    # dataset = Data.create_dataset(root='home/dataset/RoadScene', dataname='Test_vif')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                             num_workers=args.num_workers, drop_last=True, timeout=60)

    print(f'-------Starting loading {args.dataset} Dataset!-------')

    model = Spk_UNet(
       T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
       num_res_blocks=args.num_res_blocks, dropout=args.dropout, timestep=args.timestep, img_ch=args.img_ch)

    refinement_fn = FR_Net.Restormer_ROPE(in_channel=1, out_channel=1)
    ckpt_path = args.pre_trained_path 
    ckpt1 = torch.load(ckpt_path)['net_model']
    ckpt2 = torch.load(ckpt_path)['refinement_fn']
    print(f'Successfully load checkpoint!')


    model.load_state_dict(ckpt1)
    refinement_fn.load_state_dict(ckpt2)
    model.eval()
    refinement_fn.eval()

    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    for param in refinement_fn.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    sampler = GaussianDiffusionSampler(
        model, refinement_fn, float(args.beta_1), float(args.beta_T), args.T, img_size=int(args.img_size),
        mean_type=args.mean_type, var_type=args.var_type,sample_type=args.sample_type).to(device)
    if args.parallel:
        sampler = torch.nn.DataParallel(sampler)

    idx=0
    with torch.no_grad():
        result_path = '{}'.format('')
        os.makedirs(result_path, exist_ok=True)
        for _, val_data_collect in enumerate(dataloader):
            idx += 1
            name = str(val_data_collect[1][0]).replace(".jpg", "")
            val_data_collect = {key: val.to(device) for key, val in val_data_collect[0].items()}
            batch_images = sampler(val_data_collect)
            batch_images = batch_images.float().cpu()
            vis = val_data_collect['img_full'].detach().float().cpu()
            fake_single = Metrics.tensor2img(batch_images[-1].unsqueeze(0))
            img_full = Metrics.tensor2img(vis)
            fuse_img = Metrics.mergy_Y_RGB_to_YCbCr(fake_single, img_full)
            Metrics.save_img(
                fuse_img, '{}/{}.png'.format(result_path, name))


def main():
    if args.wandb:
        ## wandb init ##
        wandb.init(project="spike_diffusion", name=str(args.dataset)+str(args.sample_type))
        # suppress annoying inception_v3 initialization warning #
        warnings.simplefilter(action='ignore', category=FutureWarning)

    seed_everything(42)
    if args.train:
        train()
    if args.eval:
        eval()
    if not args.train and not args.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    # app.run(main)
    main()
