<br />
<p align="center">
  <h1 align="center">SDD-Fuse: A Multi-modality Image Fusion Framework Based on the Spiking Diffusion Fusion Model
</h1>
  <p align="center" >
    Jing Di,
    Heran Wang<sup>*</sup>,
    Jing Lian,
    Shuhui Shi,
    Jizhao Liu
  </p>


## 1. Create Environment

- Create Conda Environment
```
conda create -n diffif_env python=3.10
conda activate SDD_env
```
- Install Dependencies
```
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 2. Prepare Your Dataset

You can also refer to [MSRS](https://github.com/Linfeng-Tang/MSRS), [RoadScene](https://github.com/hanna-xu/RoadScene), [Harvard]( https://www.med.harvard.
edu/aanlib/home.html) to prepare your data. 

If you want to test only, you should list your dataset as the followed rule:
```bash
# Infrared and visible image fusion:
    home/
        dataset/
            MSRS/
                Infrared/
                Visible/
                
# Medical image fusion:
    home/
        dataset/
            Harvard/
                CT-PET-SPECT/
                MRI/
```

## 3. Training
For infrared and visible image fusion or medical image fusion train, you can use:

```shell
# Infrared and visible fusion
CUDA_VISIBLE_DEVICES=0 python main.py \
      --train \
      --dataset='MSRS, Roadscene' \
      --sample_type='ddpm' \
      --linear_start='1e-4' --linear_end='0.02' --T=1000 \
      --ch='64' \
      --ch_mult='[1 2 3 4]' \
      --timestep='4, 6, 8' \
      --img_ch='3' \
      --total_steps='20001' --multiplier='2' \
      
# Medical image fusion
CUDA_VISIBLE_DEVICES=0 python main.py \
      --train \
      --dataset='Harvard' \
      --sample_type='ddpm' \
      --linear_start='1e-4' --linear_end='0.02' --T=1000 \
      --ch='64' \
      --ch_mult='[1 2 3 4]' \
      --timestep='4, 6, 8' \
      --img_ch='3' \
      --total_steps='20001' --multiplier='2' \
```

## 4. testing
For infrared and visible image fusion or medical image fusion test, you can use:

```shell
CUDA_VISIBLE_DEVICES=0 python main.py \
    --eval \
    --pre_trained_path 'your/model' \
    --dataset='Test_mif, Test_vif' \
    --result_path='' \
```

## 5. Citation

If you find our work useful, please consider citing:

```

```













