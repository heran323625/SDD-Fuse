import math
import torch
from abc import abstractmethod
import numpy as np
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from spikingjelly.activation_based import functional, surrogate
# from spikingjelly.activation_based.neuron import LIFNode as LIFNode
from spikingjelly.activation_based.neuron import IFNode as IFNode
from timm.layers import DropPath, to_2tuple, trunc_normal_

import os



class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):  ## T: total step of diff; d_model: base channel num; dim:d_model*4
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb




class Spk_DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.neuron = IFNode(surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode='m')
        #functional.set_backend(self, backend='cupy')
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        T, B, C, H, W = x.shape
        x = self.neuron(x)
        x = x.flatten(0, 1)  ## [T*B C H W]
        x = self.conv(x)
        _, C, H, W = x.shape
        x = self.bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class Spk_UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.neuron = IFNode(surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode='m')
        #functional.set_backend(self, backend='cupy')
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        T, B, C, H, W = x.shape
        x = self.neuron(x)
        x = x.flatten(0, 1)  ## [T*B C H W]
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        _, C, H, W = x.shape
        x = self.bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class Spk_ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.neuron1 = IFNode(surrogate_function=surrogate.ATan())
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.neuron2 = IFNode(surrogate_function=surrogate.ATan())
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)


        self.in_ch = in_ch
        self.out_ch = out_ch

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = Spike_SelfAttention(out_ch)
        else:
            self.attn = nn.Identity()


        functional.set_step_mode(self, step_mode='m')
        #functional.set_backend(self, backend='cupy')
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        # init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        T, B, C, H, W = x.shape

        h = self.neuron1(x)
        h = h.flatten(0, 1)  ## [T*B C H W]
        h = self.conv1(h)
        h = self.bn1(h).reshape(T, B, -1, H, W).contiguous()

        temp = self.temb_proj(temb).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, h.shape[-2], h.shape[-1])
        h = torch.add(h, temp)

        h = self.neuron2(h)
        h = h.flatten(0, 1)  ## [T*B C H W]
        h = self.conv2(h)
        h = self.bn2(h).reshape(T, B, -1, H, W).contiguous()


        h = h + self.shortcut(x.flatten(0, 1)).reshape(T, B, -1, H, W).contiguous()

        h = self.attn(h)

        return h


class MembraneOutputLayer(nn.Module):
    """
    outputs the last time membrane potential of the LIF neuron with V_th=infty
    """

    def __init__(self, timestep=4) -> None:
        super().__init__()
        self.n_steps = timestep

    def forward(self, x):
        """
        x : (T,N,C,H,W)
        """

        arr = torch.arange(self.n_steps - 1, -1, -1)
        coef = torch.pow(0.8, arr).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(x.device)
        out = torch.sum(x * coef, dim=0)
        return out

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=4,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.neuron = IFNode(surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode='m')

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor, H, W):

        B, N, C = x.shape
        x = x.reshape(B, 1, C, H, W).contiguous()
        x = self.neuron(x)
        x = x.reshape(B*N, C).contiguous()
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class kan(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 4
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = Swish
        grid_eps = 0.02
        grid_range = [-1, 1]

        self.fc1 = KANLinear(
            in_features,
            hidden_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x, H, W)
        x = x.reshape(4, -1, N, C).contiguous()

        return x

class shiftedBlock(nn.Module):
    def __init__(self, dim,  mlp_ratio=4.,drop_path=0.,norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, dim),
        )

        self.kan = kan(in_features=dim, hidden_features=mlp_hidden_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, temb):

        temp = self.temb_proj(temb).unsqueeze(-1).unsqueeze(-1).permute(2, 0, 3, 1)
        temp = temp.expand(-1, -1, x.size(1), -1)
        x = self.drop_path(self.kan(self.norm2(x), H, W))
        x = x + temp

        return x

class D_SingleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(D_SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32,in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    def forward(self, input, temb):
        _, c, h, w = input.shape
        temp = self.temb_proj(temb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).permute(2, 0, 1,3,4)
        temp = temp.expand(-1, -1, -1, h, w)
        x = self.conv(input)
        x = x.reshape(4, -1, x.shape[1], x.shape[2], x.shape[3]).contiguous()
        x = x + temp
        x = x.flatten(0, 1)
        return x
        # return self.conv(input) + self.temb_proj(temb)[:,:,None, None].expand(input.size(0), -1, -1, -1)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
def swish(x):
    return x * torch.sigmoid(x)
import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
class Spk_UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout, timestep, img_ch=3, img_ch1=1):
        super().__init__()
        # assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'

        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)  ## T: total step of diff; ch: base channel num; tdim:ch*4
        self.timestep = timestep  ## SNN timestep
        self.conv = nn.Conv2d(img_ch, ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(ch)

        self.neuron = IFNode(surrogate_function=surrogate.ATan())
        self.conv_identity = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.bn_identity = nn.BatchNorm2d(ch)

        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(Spk_ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(Spk_DownSample(now_ch))
                chs.append(now_ch)

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(Spk_ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(Spk_UpSample(now_ch))
        assert len(chs) == 0

        self.tail_bn = nn.BatchNorm2d(now_ch)
        self.tail_swish = Swish()
        self.tail_conv = nn.Conv2d(now_ch, img_ch, kernel_size=3, stride=1, padding=1)


        self.T_output_layer = nn.Conv3d(img_ch, img_ch, kernel_size=(3,3,1), stride=(1,1,1), padding=(1,1,0))
        self.last_bn = nn.BatchNorm2d(img_ch)
        self.swish = Swish()
        self.membrane_output_layer = MembraneOutputLayer(timestep=self.timestep)
        

        functional.set_step_mode(self, step_mode='m')

        # kan
        embed_dims = [256, 320, 512]
        norm_layer = nn.LayerNorm
        dpr = [0.0, 0.0, 0.0]
        self.patch_embed3 = OverlapPatchEmbed(img_size=128 // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=128 // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])

        self.kan_block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop_path=dpr[0], norm_layer=norm_layer)])

        self.kan_block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], mlp_ratio=1, drop_path=dpr[1], norm_layer=norm_layer)])

        self.kan_dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop_path=dpr[0], norm_layer=norm_layer)])

        self.decoder1 = D_SingleConv(embed_dims[2], embed_dims[1])
        self.decoder2 = D_SingleConv(embed_dims[1], embed_dims[0])

        self.initialize()

    def initialize(self):
        # init.xavier_uniform_(self.head.weight)
        # init.zeros_(self.head.bias)
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)
        init.xavier_uniform_(self.tail_conv.weight, gain=1e-5)
        init.zeros_(self.tail_conv.bias)


    def forward(self, x, t):

        x = x.unsqueeze(0).repeat(self.timestep, 1, 1, 1, 1)  # [T, B, C, H, W]

        # Timestep embedding
        temb = self.time_embedding(t)

        # Downsampling
        T, B, C, H, W = x.shape
        h = x.flatten(0, 1)  ## [T*B C H W]
        h = self.conv(h)
        h = self.bn(h).reshape(T, B, -1, H, W).contiguous()
        h = self.neuron(h)

        h = h.flatten(0, 1)  ## [T*B C H W]
        h = self.conv_identity(h)
        h = self.bn_identity(h).reshape(T, B, -1, H, W).contiguous()

        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        h = h.flatten(0, 1)
        t3 = h
        h, H, W = self.patch_embed3(h)
        for i, blk in enumerate(self.kan_block1):
            h = blk(h, H, W, temb)
        h = h.flatten(0, 1)
        h = self.norm3(h)
        h = h.reshape(T*B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = h

        h, H, W = self.patch_embed4(h)
        for i, blk in enumerate(self.kan_block2):
            h = blk(h, H, W, temb)
        h = h.flatten(0, 1)
        h = self.norm4(h)
        h = h.reshape(T*B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4
        h = swish(F.interpolate(self.decoder1(h, temb), scale_factor=(2, 2), mode='bilinear'))

        h = torch.add(h, t4)

        _, _, H, W = h.shape
        h = h.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.kan_dblock1):
            h = blk(h, H, W, temb)

        ### Stage 3
        h = h.flatten(0, 1)
        h = self.dnorm3(h)
        h = h.reshape(T*B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        h = swish(F.interpolate(self.decoder2(h, temb), scale_factor=(2, 2), mode='bilinear'))

        h = torch.add(h, t3)
        h = h.reshape(4, -1, h.shape[1], h.shape[2], h.shape[3]).contiguous()
        for layer in self.upblocks:
            if isinstance(layer, Spk_ResBlock):
                hs1 = hs.pop()
                h = torch.cat([h, hs1], dim=2)
            h = layer(h, temb)

        T, B, C, H, W = h.shape
        h = h.flatten(0, 1)  ## [T*B C H W]
        h = self.tail_bn(h)
        h = self.tail_swish(h)
        h = self.tail_conv(h).reshape(T, B, -1, H, W).contiguous()

        h_temp = h.permute(1, 2, 3, 4, 0)  # [ B, C, H, W, T]
        h_temp = self.T_output_layer(h_temp).permute(4, 0, 1, 2, 3)   # [ T, B, C, H, W]
        h_temp = self.last_bn(h_temp.flatten(0,1)).reshape(T, B, -1, H, W).contiguous()
        h = self.swish(h_temp) + h  # [ T, B, C, H, W]

        h = self.membrane_output_layer(h)


        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 2
    model = Spk_UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 4], attn=[8],
        num_res_blocks=2, dropout=0.1, timestep=4).cuda()

    ## Load model
    # ckpt = torch.load(os.path.join('/home/jiahang/jiahang/Diffusion_with_spk/pytorch-ddpm/logs/threshold_test', 'snnbest.pt'))
    # model.load_state_dict(ckpt['net_model'])

    ckpt = torch.load(os.path.join('D:\conda\envs/two_paper\Python_work\SDM-master\SDM-master\SDM\SDM_TSM_8T.pt'))
    model.load_state_dict(ckpt['net_model'])

    x = torch.randn(batch_size, 3, 32, 32).cuda()
    t = torch.randint(1000, (batch_size,)).cuda()
    # print(model)
    y = model(x, t)
    print(y.shape)
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    functional.reset_net(model)
