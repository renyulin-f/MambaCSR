import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
import logging

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn_no_out_proj
except ImportError:
    mamba_inner_fn_no_out_proj = None

from .local_scan import local_scan, local_scan_bchw, local_reverse, cross_scale_reverse, cross_scale_scan, reverse_nested_local_scan_bchw

class MultiScan(nn.Module):
    ALL_CHOICES = ('w8', 'w8_flip','c_w8','c_w8_flip') #Not use in MambaCSR
    def __init__(self, dim, choices=None, token_size=(14, 14)):
        super().__init__()
        if choices is None:
            self.choices = MultiScan.ALL_CHOICES
            self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.choices])
            self.weights = nn.Parameter(1e-3 * torch.randn(len(self.choices), 1, 1, 1))
            self._iter = 0
            self.logger = logging.getLogger()
            self.search = True
        else:
            self.choices = choices
            self.search = False

    def forward(self, xs):
        """
        Input @xs: [[B, L, D], ...]
        """
        if self.search:
            weights = self.weights.softmax(0)
            xs = [norm(x) for norm, x in zip(self.norms, xs)]
            xs = torch.stack(xs) * weights
            x = xs.sum(0)
            if self._iter % 200 == 0:
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    self.logger.info(str(weights.detach().view(-1).tolist()))
            self._iter += 1
        else:
            x = torch.stack(xs).sum(0)
        return x

    def multi_scan(self, x):
        """
        Input @x: shape [B, L, D]
        """
        xs = []
        for direction in self.choices:
            xs.append(self.scan(x, direction)) #(2,192,64,64)
        return xs

    def multi_reverse(self, xs):
        new_xs = []
        for x, direction in zip(xs, self.choices):
            new_xs.append(self.reverse(x, direction))
        return new_xs
    
    def cross_scale_scan(self, x,x_down):
        """
        Input @x: shape [B, L, D]
        """
        xs = []
        for direction in self.choices:
            xs.append(self.cross_scan_1(x, x_down,direction)) #(2,192,64,64)
        return xs
    
    def cross_reverse(self, xs):
        new_xs = []
        for x, direction in zip(xs, self.choices):
            new_xs.append(self.cross_reverse_1(x, direction))
        return new_xs    
        
    def scan(self, x, direction='h'):
        """
        Input @x: shape [B, L, D] or [B, C, H, W]
        Return torch.Tensor: shape [B, D, L]
        """
        B,C,H,W = x.shape
        self.token_size = (H,W)
        if len(x.shape) == 3:
            if direction == 'h':
                return x.transpose(-2, -1)
            elif direction == 'h_flip':
                return x.transpose(-2, -1).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('w'):
                K = int(direction[1:].split('_')[0])
                flip = direction.endswith('flip')
                return local_scan(x, K, H, W, flip=flip)
            else:
                raise RuntimeError(f'Direction {direction} not found.')
            
        elif len(x.shape) == 4:
            if direction == 'h':
                return x.flatten(2)
            elif direction == 'h_flip':
                return x.flatten(2).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('w'):
                K = int(direction[1:].split('_')[0])
                flip = direction.endswith('flip')
                return local_scan_bchw(x, K, H, W, flip=flip)
            elif direction.startswith('c'):
                K = int(direction[3:].split('_')[0])
                flip = direction.endswith('flip')
                return local_scan_bchw(x, K, H, W, flip=flip,column_first=True)     
            else:
                raise RuntimeError(f'Direction {direction} not found.')

    def reverse(self, x, direction='h'):
        """
        Input @x: shape [B, D, L]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.token_size
        if direction == 'h':
            return x
        elif direction == 'h_flip':
            return x.flip([-1])
        elif direction == 'v':
            return rearrange(x, 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction == 'v_flip':
            return rearrange(x.flip([-1]), 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction.startswith('w'):
            K = int(direction[1:].split('_')[0])
            flip = direction.endswith('flip')
            return local_reverse(x, K, H, W, flip=flip)
        elif direction.startswith('c'):
            K = K = int(direction[3:].split('_')[0])
            flip = direction.endswith('flip')
            return local_reverse(x, K, H, W, flip=flip,column_first=True)
        else:
            raise RuntimeError(f'Direction {direction} not found.')    
    
    def cross_scan_1(self, x,x_down, direction='h'):
        """
        Input @x: shape [B, L, D] or [B, C, H, W]
        Return torch.Tensor: shape [B, D, L]
        """
        B,C,H,W = x.shape
        self.token_size = (H,W)
        if direction.startswith('w'):
            K = int(direction[1:].split('_')[0])
            flip = direction.endswith('flip')
            return cross_scale_scan(x, x_down, w1=K, flip=flip, column_first=False)
        elif direction.startswith('c'):
            K = int(direction[3:].split('_')[0])
            flip = direction.endswith('flip')
            return cross_scale_scan(x, x_down, w1=K, flip=flip, column_first=True)
        else:
            raise RuntimeError(f'Direction {direction} not found.')
        
    def cross_reverse_1(self, x, direction='h'):
        """
        Input @x: shape [B, D, L]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.token_size
        # x = self.linear(x)
        if direction.startswith('w'):
            K = int(direction[1:].split('_')[0])
            flip = direction.endswith('flip')
            return cross_scale_reverse(x, H, W, w1=K, flip=flip, column_first=False)
        elif direction.startswith('c'):
            K = K = int(direction[3:].split('_')[0])
            flip = direction.endswith('flip')
            return cross_scale_reverse(x, H, W, w1=K, flip=flip, column_first=True)
        else:
            raise RuntimeError(f'Direction {direction} not found.') 
        
    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace(self.__class__.__name__, f'{self.__class__.__name__}[{scans}]')


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        attn = c_attn #* s_attn  # [B, N, C]
        return ori_x * attn


class MultiMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        directions=None,
        token_size=(14, 14),
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.token_size = token_size

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.activation = "silu"
        self.act = nn.SiLU()


        self.multi_scan = MultiScan(self.d_inner, choices=directions, token_size=token_size)
        '''new for search'''
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        for i in range(len(self.multi_scan.choices)):
            setattr(self, f'A_log_{i}', nn.Parameter(A_log))
            getattr(self, f'A_log_{i}')._no_weight_decay = True

            conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            setattr(self, f'conv1d_{i}', conv1d)

            x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            setattr(self, f'x_proj_{i}', x_proj)

            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

            setattr(self, f'dt_proj_{i}', dt_proj)

            D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            D._no_weight_decay = True
            setattr(self, f'D_{i}', D)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.attn = BiAttn(self.d_inner)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        xz = self.in_proj(hidden_states)

        xs = self.multi_scan.multi_scan(xz)  # [[BDL], [BDL], ...]
        outs = []
        for i, xz in enumerate(xs):
            # xz = rearrange(xz, "b l d -> b d l")
            A = -torch.exp(getattr(self, f'A_log_{i}').float())
            conv1d = getattr(self, f'conv1d_{i}')
            x_proj = getattr(self, f'x_proj_{i}')
            dt_proj = getattr(self, f'dt_proj_{i}')
            D = getattr(self, f'D_{i}')
            
            out = mamba_inner_fn_no_out_proj(
                xz,
                conv1d.weight,
                conv1d.bias,
                x_proj.weight,
                dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                D,
                delta_bias=dt_proj.bias.float(),
                delta_softplus=True,
            )
            outs.append(out)

        outs = self.multi_scan.multi_reverse(outs)
        outs = [self.attn(rearrange(out, 'b d l -> b l d')) for out in outs]
        out = self.multi_scan(outs)
        out = F.linear(out, self.out_proj.weight, self.out_proj.bias)

        return out
