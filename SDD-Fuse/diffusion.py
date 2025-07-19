import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import matplotlib.pyplot as plt
import numpy as np
from loss import Grad_loss
from loss import SSIM_loss
from tqdm import tqdm

device = torch.device('cuda:0')

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def extract2(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

        
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, refinement_fn, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.refinement_fn = refinement_fn
        self.T = T
        self.ddim_eta = 1
        self.grad_loss = Grad_loss()
        self.ssim_loss = SSIM_loss()

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'alphas_cumprod', alphas_bar)
        self.register_buffer(
            'alphas_cumprod_prev', alphas_bar_prev)
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t - \
               extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * noise

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        x_start = x_0['fusion']
        x_start_vis = x_0['vis']
        x_start_ir = x_0['ir']
        [b, _, _, _] = x_start_vis.shape

        t = torch.randint(1, self.T, size=(b,), device=x_start.device)

        noise = torch.randn_like(x_start)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_bar, t, x_start.shape) * noise)

        pred_noise=self.model(torch.cat((x_start_vis, x_start_ir, x_t), dim=1), t)
        pred_noise = pred_noise.mean(dim=1, keepdim=True)

        loss_eps = 8 * F.mse_loss(pred_noise, noise, reduction='none')
        pred_x0 = self.predict_start_from_noise(x_t, t-1, pred_noise)
        pred_x0.clamp_(-1., 1.)
        pred_x0_detach = pred_x0.detach()
        refine_x0 = self.refinement_fn(torch.cat([x_start_vis, x_start_ir], 1), pred_x0_detach, t)

        refine_x0.clamp_(-1., 1.)

        max_img = torch.max(x_start_vis, x_start_ir)
        loss_max = 4 * F.mse_loss(refine_x0, max_img)
        loss_grad = 5 * self.grad_loss(refine_x0, x_start_vis, x_start_ir)
        loss_ssim = self.ssim_loss(refine_x0, x_start_vis) + self.ssim_loss(refine_x0, x_start_ir)
        loss_simple = loss_max + loss_grad + loss_ssim
        loss_x0 = 2 * F.mse_loss(refine_x0, x_start)

        # loss_x0 = 2 * F.mse_loss(x_prev, x_t1, reduction='none')
        loss = (loss_eps + loss_x0 + loss_simple)/20
        # loss = (loss_eps + loss_simple) / 20
        return loss

    

class GaussianDiffusionLogger(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    
    def forward(self, x_0):
        '''
        Evaluate the loss through time
        '''

        print(f'{x_0.shape[0]} images start computing loss through time')
        t_list = torch.linspace(start=0, end=self.T-1, steps=self.T, dtype=torch.int64, device=x_0.device)
        # t_list = t_list.view(len(t_list))
        loss_list = []
        with torch.no_grad():
            for t in t_list:
                t = t.unsqueeze(0).repeat(x_0.shape[0])
                noise = torch.randn_like(x_0)
                x_t = (
                    extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                    extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
                loss = F.mse_loss(self.model(x_t, t), noise, reduction='mean')
                loss_list.append(loss.item())

                functional.reset_net(self.model)
        
        return t_list.cpu().numpy(), loss_list
        # fig = plt.figure()
        # plt.plot(t_list.cpu().numpy(),loss_list)
        # plt.title('Loss through Time')
        # plt.xlabel('t')
        # plt.ylabel('loss')

        # return fig



class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, refinement_fn, beta_1, beta_T, T, img_size=32,
                 mean_type='xstart', var_type='fixedlarge',sample_type='ddpm'):
        print(mean_type)
        assert mean_type in ['xprev','xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        assert sample_type in ['ddpm', 'ddim','ddpm2']
        super().__init__()

        self.model = model
        self.refinement_fn = refinement_fn
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.sample_type = sample_type

        # beta_t
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        # alpha_t
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'one_minus_alphas_bar', (1.- alphas_bar))
        self.register_buffer(
            'sqrt_recip_one_minus_alphas_bar', 1./torch.sqrt(1.- alphas_bar))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        #print((extract(self.sqrt_recip_alphas_bar, t, x_t.shape)).dtype)
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, img, x_e):
        # below: only log_variance is used in the KL computations
        # Mean parameterization
        if self.sample_type=='ddpm':
            model_log_var = {
                # for fixedlarge, we set the initial (log-)variance like so to
                # get a better decoder log likelihood
                'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                                self.betas[1:]])),
                'fixedsmall': self.posterior_log_var_clipped,
            }[self.var_type]
            model_log_var = extract(model_log_var, t, img.shape)
            if self.mean_type == 'xprev':       # the model predicts x_{t-1}
                x_prev = self.model(x_t, t)
                x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
                model_mean = x_prev
            elif self.mean_type == 'xstart':    # the model predicts x_0
                x_0 = self.model(x_t, t)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t)
            elif self.mean_type == 'epsilon':   # the model predicts epsilon
                eps = self.model(x_t, t)
                eps = eps.mean(dim=1, keepdim=True)
                x_0 = self.predict_xstart_from_eps(img, t, eps=eps)
                #print(x_0.dtype)
                x_0 = x_0.clamp(-1.,1.)
                pred_x0_detach = x_0.detach()
                refine_x0 = self.refinement_fn(x_e, pred_x0_detach, t)
                refine_x0.clamp_(-1., 1.)
                model_mean, _ = self.q_mean_variance(refine_x0, img, t)
            else:
                raise NotImplementedError(self.mean_type)
            #(model_mean)
            x_0 = torch.clip(x_0, -1., 1.)
            
            functional.reset_net(self.model)

            return model_mean, model_log_var
        elif self.sample_type=='ddim':
            eps = self.model(x_t, t)
            a_t  = extract(self.sqrt_alphas_bar, t, x_t.shape)
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
            a_s  = extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)

            a_ts = a_t/a_s
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2

            x0_t = (x_t - eps*sigma_t)/(a_t)
            x0_t = x0_t.clamp(-1.,1.)
            eta = 0
            c_1 = eta * torch.sqrt((1-a_t.pow(2)/a_s.pow(2)) * (1-a_s.pow(2))/(1-a_t.pow(2)))
            c_2 = torch.sqrt((1-a_s.pow(2))-c_1.pow(2))
            mean = a_s * x0_t + c_2*eps + c_1 * torch.randn_like(x_t)
            functional.reset_net(self.model)
            return mean
        elif self.sample_type=='ddpm2':
            model_log_var = {
                # for fixedlarge, we set the initial (log-)variance like so to
                # get a better decoder log likelihood
                'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                                self.betas[1:]])),
                'fixedsmall': self.posterior_log_var_clipped,
            }[self.var_type]
            model_log_var = extract(model_log_var, t, x_t.shape)

            eps = self.model(x_t, t)

            a_t  = extract2(self.sqrt_alphas_bar, t, x_t.shape)
            a_s  = extract2(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
            sigma_t = torch.sqrt(extract2(self.one_minus_alphas_bar, t, x_t.shape))
            sigma_s = torch.sqrt(extract2(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
            a_ts = a_t/a_s
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2
            mean_x0 = ((1/a_t).float()*x_t - (sigma_t/a_t).float() * eps)
            mean_x0 = mean_x0.clamp(-1.,1.)
            mean_xs = (a_ts*sigma_s.pow(2)/(sigma_t.pow(2))).float() * x_t + (a_s*beta_ts/(sigma_t.pow(2))).float() * mean_x0

            functional.reset_net(self.model)
            return mean_xs, model_log_var
        else:
            pass

    def forward(self, x_T):
        xt = x_T['vis']
        shape = xt.shape
        b, c, h, w = x_T['vis'].shape
        img = torch.randn(shape, device=device)
        ret_img = xt
        x_t = torch.cat([x_T['vis'], x_T['ir'], img], dim=1)
        c = self.T // 4
        ddim_timestep_seq = list(reversed(range(self.T-1, -1, -c)))
        ddim_timestep_seq = np.asarray(ddim_timestep_seq)

        for i in tqdm(reversed(range(0, 4)), desc='sampling loop time step',
                      total=4):
            t = x_t.new_ones([b, ], dtype=torch.long)  * ddim_timestep_seq[i]
            if self.sample_type =='ddpm' or self.sample_type =='ddpm2':
                #print(x_t.dtype)
                # no noise when t == 0
                x_t = torch.cat([x_T['vis'], x_T['ir'], img], dim=1)
                x_e = torch.cat([x_T['vis'], x_T['ir']], 1)
                if i > 0:
                    mean, log_var = self.p_mean_variance(x_t=x_t, t=t, img=img, x_e=x_e)
                    noise = torch.randn_like(img)
                    img = mean + torch.exp(0.5 * log_var) * noise
                    ret_img = torch.cat([ret_img, img], dim=0)
                else:
                    eps = self.model(x_t, t)
                    eps = eps.mean(dim=1, keepdim=True)
                    a_ts = extract(self.sqrt_alphas_bar, t, img.shape)
                    sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, img.shape))
                    beta_ts = (1-a_ts**2)
                    x_0 = 1/a_ts*( img - eps * beta_ts/sigma_t)
                    x_0 = torch.clip(x_0, -1, 1)
                    pred_x0_detach = x_0.detach()
                    refine_x0 = self.refinement_fn(x_e, pred_x0_detach, t)
                    ret_img = torch.cat([ret_img, refine_x0], dim=0)

            else:
                if i == 0: return x_t
                x_t = self.p_mean_variance(x_t=x_t, t=t)
        return ret_img.detach()



