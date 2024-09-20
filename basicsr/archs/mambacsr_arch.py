# Code Implementation of the MambaCSR Model
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Callable
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import repeat
import sys
import copy
from fvcore.nn import flop_count
sys.path.append('basicsr/archs')
from mamba.multi_mamba import MultiScan

from csms6s import selective_scan_flop_jit
try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    if torch.__version__ > '2.0.0':
        from selective_scan_vmamba_pt202 import selective_scan_cuda_core
    else:
        from selective_scan_vmamba import selective_scan_cuda_core
except Exception as e:
    # print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    
# import selective_scan_cuda

NEG_INF = -1000000

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y
    
class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class SelectiveScan(torch.autograd.Function):
    
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        
        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
        #out = (1536,4900)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
        
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


class MultiScanVSSM(MultiScan):
    def __init__(self, dim, choices=None):
        super().__init__(dim, choices=choices, token_size=None)
        self.attn = BiAttn(dim)
    def merge(self, xs):
        # remove the padded tokens
        xs = [xs[:, i, :, :l] for i, l in enumerate(self.scan_lengths)]
        xs = super().multi_reverse(xs)
        xs = [self.attn(x.transpose(-2, -1)) for x in xs]
        x = super().forward(xs)
        return x    
    
    def multi_scan(self, x):
        B, C, H, W = x.shape #(B,192,64,64)
        self.token_size = (H, W)
        xs = super().multi_scan(x)  # [[B, C, L], ...]
        self.scan_lengths = [x.shape[2] for x in xs]
        max_length = max(self.scan_lengths)
        new_xs = []
        for x in xs: #(2,192,4096)
            if x.shape[2] < max_length:
                x = F.pad(x, (0, max_length - x.shape[2]))
            new_xs.append(x)
        return torch.stack(new_xs, 1)
    
    def cross_merge(self, xs):
        # remove the padded tokens
        xs = [xs[:, i, :, :l] for i, l in enumerate(self.scan_lengths)]
        xs = super().cross_reverse(xs)
        xs = [self.attn(x.transpose(-2, -1)) for x in xs]
        x = super().forward(xs)
        return x    
    
    def cross_scale_scan(self, x, x_down):
        B, C, H, W = x.shape #(B,192,64,64)
        self.token_size = (H, W)
        xs = super().cross_scale_scan(x,x_down)  # [[B, C, L], ...]
        self.scan_lengths = [x.shape[2] for x in xs]
        max_length = max(self.scan_lengths)
        # pad the tokens into the same length as VMamba compute all directions together
        new_xs = []
        for x in xs: #(2,192,4096)
            if x.shape[2] < max_length:
                x = F.pad(x, (0, max_length - x.shape[2]))
            new_xs.append(x)
        return torch.stack(new_xs, 1)

    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace('MultiScanVSSM', f'MultiScanVSSM[{scans}]')


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        attn = c_attn #* s_attn  # [B, N, C]
        out = ori_x * attn
        return out


def multi_selective_scan_cross(
    x: torch.Tensor=None, 
    x_down=None,
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    nrows = -1,
    delta_softplus = True,
    to_dtype=True,
    multi_scan=None,
):
    B, D, H, W = x.shape #B, C, 64, 64
    D, N = A_logs.shape #N=16
    K, D, R = dt_projs_weight.shape #K=8, D=192, R=12   R=rank=12
    L = H * W #64*64   4096

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
    
    xs = multi_scan.cross_scale_scan(x,x_down) 
    L = xs.shape[-1]
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight) # l fixed
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)
    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, L) #(out= 2,8,192,4900)
    y = multi_scan.cross_merge(ys)  #(out= 2,4096,192)
    y = out_norm(y).view(B, H, W, -1)
    return (y.to(x.dtype) if to_dtype else y)


def multi_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    nrows = -1,
    delta_softplus = True,
    to_dtype=True,
    multi_scan=None,
):
    B, D, H, W = x.shape #2, 192, 64, 64
    D, N = A_logs.shape #N=16
    K, D, R = dt_projs_weight.shape #K=8, D=192, R=12   R代表rank=12
    L = H * W #64*64   4096

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
    
    xs = multi_scan.multi_scan(x) #(2,8,192,4900)
    L = xs.shape[-1]
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight) # l fixed

    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    
    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)
    
    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, L)
    y = multi_scan.merge(ys)
    y = out_norm(y).view(B, H, W, -1)
    return (y.to(x.dtype) if to_dtype else y)


class LW_SS2D(nn.Module): #Local-Window Based 
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        directions=None,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model) #360
        d_inner = d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank #dt_rank=12
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv # =3
    #nn.LayerNorm
        self.out_norm = nn.LayerNorm(d_inner) #360
        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs) #180->720
        self.act: nn.Module = act_layer() #SiLU()
        self.K = len(MultiScanVSSM.ALL_CHOICES) if directions is None else len(directions)
        self.K2 =self.K
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand, #360
                out_channels=d_expand, #360
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) #(360,44)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N) (1440,16)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # Local Mamba
        self.multi_scan = MultiScanVSSM(d_expand, choices=directions)

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner))) 

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, nrows=-1, channel_first=False):
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = multi_selective_scan( #Input=(B,192,64,64)
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, self.out_norm,
            nrows=nrows, delta_softplus=True, multi_scan=self.multi_scan,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor):
        xz = self.in_proj(x) #(B,64,64,96)-> (B,64,64,384)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            z = self.act(z)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x)) # (2,192,64,64)
        else:
            xz = self.act(xz)
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
        y = self.forward_core(x, channel_first=(self.d_conv > 1)) #input_x=(1,360,48,48)
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class RLMBlock(nn.Module):  # Residual local mamba block (RLMB block)
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            d_state: int = 16,
            mlp_ratio: float = 2.,
            dual_interleaved_scan = False,
            directions = [],
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.dual_interleaved_scan = dual_interleaved_scan
        #Attention Part: (Mamba for modeling contextual information)
        self.self_attention = LW_SS2D(d_model=hidden_dim,d_state=d_state,directions=directions) #默认窗口大小为8,并且每个窗口内4scan方式
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.drop_path = DropPath(drop_path)
        #FFN Part:
        self.conv_block = CAB(num_feat=hidden_dim, compress_ratio=24, squeeze_factor=24)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_ratio*hidden_dim, act_layer=nn.GELU, drop=0.)
      
    def forward(self, input, x_size):
        # x [B,HW,C]
        B, _, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        conv_x = self.conv_block(x.permute(0, 3, 1, 2).contiguous())
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, *x_size, C)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x)) + conv_x * 0.01
        x = x * self.skip_scale2 + self.drop_path(self.mlp(self.ln_2(x)))
        x = x.view(B, -1, C).contiguous()
        return x


class Cross_SS2D(nn.Module): # Cross-scale Module: Modeling multi-scale contexual information.
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        directions=None,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model) #360
        d_inner = d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank #dt_rank=12
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv # =3
        self.out_norm = nn.LayerNorm(d_inner) #360
        self.in_proj = nn.Linear(d_model, d_expand *2, bias=bias, **factory_kwargs) #180->720
        self.in_proj_down= nn.Linear(d_model, d_expand *2, bias=bias, **factory_kwargs) #180->720
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.act: nn.Module = act_layer() #SiLU()
        self.act_down: nn.Module = act_layer() #SiLU()
        self.K = len(MultiScanVSSM.ALL_CHOICES) if directions is None else len(directions)
        self.K2 =self.K
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand, #360
                out_channels=d_expand, #360
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d2 = nn.Conv2d(
                in_channels=d_expand, #360
                out_channels=d_expand, #360
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)
        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) #(360,44)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N) (1440,16)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)
        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        # Local Mamba
        self.multi_scan = MultiScanVSSM(d_expand, choices=directions)
        self.mlp = Mlp(in_features=d_model, hidden_features=4*d_model, act_layer=nn.GELU, drop=0.)
        self.skip_scale = nn.Parameter(torch.ones(d_model))
        self.skip_scale2 = nn.Parameter(torch.ones(d_model))
        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner))) 

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def forward_core(self, x: torch.Tensor, x_down,nrows=-1,channel_first=False):
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = multi_selective_scan_cross( #Input=(B,192,64,64)
            x, x_down,self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, self.out_norm,
            nrows=nrows, delta_softplus=True, multi_scan=self.multi_scan,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x
    
    def forward(self, x, x_down,x_size):
        B, L, C = x.shape
        x = x.view(B, *x_size, C).contiguous()  # [B,H,W,C]  
        input = x
        x = self.ln_1(x)
        xz = self.in_proj(x) #(B,64,64,C)-> (B,64,64,4*C)
        
        x_down_size = (x_size[0]//2,x_size[1]//2)
        x_down = x_down.view(B, *x_down_size, C).contiguous()
        xz_down = self.in_proj_down(x_down)

        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) 
        
        x_down, z_down = xz_down.chunk(2, dim=-1) # (b, h, w, d)
        z_down = self.act_down(z_down)
        x_down = x_down.permute(0, 3, 1, 2).contiguous()
        x_down = self.act_down(self.conv2d2(x_down)) 

        y = self.forward_core(x,x_down,channel_first=(self.d_conv > 1)) 
        y = y * z
        out = self.dropout(self.out_proj(y))
        out = input * self.skip_scale + out
        out = out * self.skip_scale2 + self.mlp(self.ln_2(out))
        out = out.view(B, -1, C).contiguous()
        
        return out


class BasicLayer(nn.Module):
    """ The Basic MambaCSR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        dual_interleaved_scan (bool): Whether to use dual-interleaved scanning method.
        cross_scale_scan (bool): Whether to use cross-scale scanning method.
        
    """
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 dual_interleaved_scan = False,
                 cross_scale_scan=False,
                 scan_size = 8):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint
        if cross_scale_scan:
            if depth < 6: # down-scale image go sthrough only one RLMB block
                directions = ['w32','w32_flip']
                directions2 = ['c_w32','c_w32_flip']
                directions3 = ['w32','w32_flip']
                directions4 = ['c_w32','c_w32_flip']
            else:
                directions = ['w'+str(scan_size),'w'+str(scan_size)+'_flip']  # 
                directions2 = ['c_w'+str(scan_size),'c_w'+str(scan_size)+'_flip']
                directions3 = ['w64','w64_flip']
                directions4 = ['c_w64','c_w64_flip']
        elif dual_interleaved_scan:
                directions = ['w'+str(scan_size),'w'+str(scan_size)+'_flip']
                directions2 = ['c_w'+str(scan_size),'c_w'+str(scan_size)+'_flip']
                directions3 = ['w64','w64_flip']
                directions4 = ['c_w64','c_w64_flip']
        else:
            directions = ['w'+str(scan_size),'w'+str(scan_size)+'_flip', 'c_w'+str(scan_size),'c_w'+str(scan_size)+'_flip']
        # build blocks
        self.blocks = nn.ModuleList()
        if cross_scale_scan:
            for i in range(depth):
                self.blocks.append(RLMBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=nn.LayerNorm,
                    d_state=d_state,
                    mlp_ratio=self.mlp_ratio,
                    input_resolution=input_resolution,
                    dual_interleaved_scan = dual_interleaved_scan,
                    directions=(directions if i % 4 == 0 else 
                                directions2 if i % 4 == 1 else 
                                directions3 if i % 4 == 2 else 
                                directions4)
                    ))
        elif dual_interleaved_scan:
            for i in range(depth):
                self.blocks.append(RLMBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=nn.LayerNorm,
                    d_state=d_state,
                    mlp_ratio=self.mlp_ratio,
                    input_resolution=input_resolution,
                    dual_interleaved_scan = dual_interleaved_scan,
                    directions=(directions if i % 4 == 0 else 
                                directions2 if i % 4 == 1 else 
                                directions3 if i % 4 == 2 else 
                                directions4)
                    ))
        else:
            for i in range(depth):
                self.blocks.append(RLMBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=nn.LayerNorm,
                    d_state=d_state,
                    mlp_ratio=self.mlp_ratio,
                    input_resolution=input_resolution,
                    dual_interleaved_scan = dual_interleaved_scan,
                    directions=directions
                    ))
            
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    
    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


class ResidualLocalGroup(nn.Module):
    """Residual Local Mamba Group (RLMG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
        dual_interleaved_scan: Whether use Dual-Interleaved scanning method.
        cross_scale_scan: Whether use Cross-scale scanning method.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 d_state = 16,
                 mlp_ratio = 2.,
                 drop_path = 0.,
                 norm_layer = nn.LayerNorm,
                 downsample = None,
                 use_checkpoint = False,
                 img_size = None,
                 patch_size = None,
                 resi_connection = '1conv',
                 dual_interleaved_scan = False,
                 cross_scale_scan = False,
                 scan_size = 8):
        super(ResidualLocalGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            dual_interleaved_scan = dual_interleaved_scan,
            cross_scale_scan = cross_scale_scan,
            scan_size = scan_size)

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


@ARCH_REGISTRY.register()
class MambaCSR(nn.Module):
    r""" MambaCSR Model
           A PyTorch impl of : `MambaCSR: Dual-Interleaved Scanning for Compressed Image Super-Resolution With SSMs`.

       Args:
           img_size (int | tuple(int)): Input image size. Default: 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           upscale: Upscale factor. Default: 4
           embed_dim (int): Patch embedding dimension. Default: 180
           d_state (int): num of hidden state in the state space model. Default: 16
           depths (tuple(int)): Depth of each RLMG
           depths2: Depth of Cross-Scale Module
           mlp_ratio: Defualt: 2
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
           dual_interleaved_scan: Default: True,
           cross_scale_scan: Default: False
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           
           
       """
    def __init__(self,
                 img_size = 64,
                 patch_size = 1,
                 in_chans = 3,
                 upscale = 4,
                 embed_dim = 180,
                 d_state = 16,
                 depths = (6, 6, 6, 6, 6, 6),
                 depths2 = (6),
                 mlp_ratio = 2.,
                 img_range = 1.,
                 upsampler = '',
                 resi_connection = '1conv',
                 dual_interleaved_scan = True,
                 cross_scale_scan = False,
                 scan_size = 8,
                 drop_rate = 0.,
                 drop_path_rate = 0.1,
                 norm_layer = nn.LayerNorm,
                 patch_norm = True,
                 use_checkpoint = False,
                 **kwargs):
        super(MambaCSR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.cross_scale_scan = cross_scale_scan
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio # 2
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1) #(3,180,)
        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths) #=6
        self.embed_dim = embed_dim #180  我有时候直接改成96
        self.patch_norm = patch_norm # path_norm = True
        self.num_features = embed_dim # 180
        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution   
        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)  # 这个是?
         # Here define the module for cross-scale scanning
        if cross_scale_scan:
            self.pos_drop2 = nn.Dropout(p=drop_rate)
            self.num_layers2 = len(depths2) 
            self.patch_embed2 = PatchEmbed(
                img_size=img_size//2,
                patch_size=patch_size,
                in_chans=embed_dim,
                embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
        
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 6-layer
            layer = ResidualLocalGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state = d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                dual_interleaved_scan = dual_interleaved_scan,
                cross_scale_scan=cross_scale_scan,
                scan_size = scan_size
            )
            self.layers.append(layer)
            
        if cross_scale_scan:
            self.layers2 = nn.ModuleList()
            self.length = len(depths2)
            for i_layer in range(self.num_layers2): # 1-layer
                layer = ResidualLocalGroup(
                    dim=embed_dim,
                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                    depth=depths2[i_layer],
                    d_state = d_state,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=use_checkpoint,
                    img_size=img_size,
                    patch_size=patch_size,
                    resi_connection=resi_connection,
                    dual_interleaved_scan = dual_interleaved_scan,
                    cross_scale_scan = cross_scale_scan,
                )
                self.layers2.append(layer)
            self.layers3 = nn.ModuleList() # Fusion Layer, cross-scale scanning
            for _ in range(0,len(depths2)):
                layer = Cross_SS2D(d_model=embed_dim,directions=['w16','w16_flip','c_w16','c_w16_flip'])
                self.layers3.append(layer)
                
        self.norm = norm_layer(self.num_features)
        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
        # -------------------------3. high-quality image reconstruction ------------------------ #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        if self.cross_scale_scan:
            x_down = F.interpolate(x, mode='bilinear', scale_factor=0.5, align_corners=False)
            x_down_size = (x_down.shape[2],x_down.shape[3])
            x_down = self.patch_embed2(x_down) # N,L,C
            x_down = self.pos_drop2(x_down) #(B,L,C)
        x = self.patch_embed(x) # N,L,C
        x = self.pos_drop(x) #(B,L,C)
        if self.cross_scale_scan:
            index = 0
            for index, layer1 in enumerate(self.layers):
                x = layer1(x, x_size)
                if index < self.length:
                    # 在前两个循环时执行 layer2 和 layer3
                    x_down = self.layers2[index](x_down, x_down_size)
                    x = self.layers3[index](x, x_down, x_size)   
        else:
            for layer in self.layers:
                x = layer(x, x_size)
        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        x = x / self.img_range + self.mean

        return x

    def flops(self, shape=(3, 64, 64)):
        model = copy.deepcopy(self)
        model.cuda().eval()
        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        supported_ops = {
            "aten::silu": None,     # 64*64*3 = 12288
            "aten::sub": None,      # Could be ignored
            "aten::mul": None,      # 12288 * 2, could be ignored
            "prim::PythonOp.SelectiveScanMamba": partial(selective_scan_flop_jit),
        }
        Gflops, _ = flop_count(inputs=(input,), model=model,supported_ops=supported_ops)
        del model, input
        return sum(Gflops.values()) * 1e9

