from encodings.punycode import T
import math
import copy
from functools import partial
from typing import Optional, Callable

import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from einops import rearrange, repeat
from timm.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import networks.mamba_sys as mamba_sys

# from pytorch_model_summary import summary
from torchsummary import summary

class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, groups=1, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, 
                              embed_dim, 
                              kernel_size=patch_size,
                              stride=patch_size,
                              groups=groups)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    
    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class PatchMerging2D(nn.Module):
    r""" PatchMerging2D performs spatial downsampling by a factor of 2.
        It groups each 2×2 neighborhood of tokens, concatenates their channels
        (C → 4C), applies LayerNorm, and projects them to a lower-dimensional
        embedding (4C → 2C). The output has half the spatial resolution and
        twice the channel dimension: (B, H, W, C) → (B, H/2, W/2, 2C).
        Patch Merging Layer.
    Args:
        dim (int): Resolution of input token.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if H % 2 != 0:
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        
        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = tc.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    
    
class PatchExpand(nn.Module):
    """
    PatchExpand layer.

    Upsamples the input feature map by a factor of 2 by converting channel
    information into spatial resolution. The operation applies a linear
    projection followed by a PixelShuffle-style rearrangement:
    (B, H, W, C) → (B, 2H, 2W, C/2).
    """
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(
            dim,  2*dim, bias = False) if dim_scale == 2 else nn.Identity()
            # applied to last dimension
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = self.norm(x)
        return x
    
class FinalPatchExpand_X4(nn.Module):
    """
    FinalPatchExpand_X4 layer.

    Upsamples the input feature map by a factor of 4 by converting channel
    information into spatial resolution. The operation applies a linear
    projection followed by a PixelShuffle-style rearrangement:
    (B, H, W, C) → (B, 4H, 4W, C).
    """
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias = False)
            # applied to last dimension
        self.output_dim = dim            
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=4, p2=4, c=C//16)
        x = self.norm(x)
        return x    
    
class SS2D(nn.Module):
    r""" Mamba SSM layer for 2D inputs.
    2D-Selective-Scan for Vision Data (SS2D)
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv 
        self.expand = expand
        self.d_inner = int(self.d_model * self.expand)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection to xz
        self.in_proj = nn.Linear(self.d_model,
                                 self.d_inner*2,
                                 bias=bias,
                                 **factory_kwargs)
        # DW conv
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv-1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # 4 projections for [dt, B, C] calculation
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank+self.d_state*2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank+self.d_state*2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank+self.d_state*2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank+self.d_state*2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(tc.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(tc.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(tc.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.forward_core = self.forward_corev0
        
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", 
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
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
        dt = tc.exp(
            tc.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + tc.log(-tc.expm1(-dt))
        with tc.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            tc.arange(1, d_state + 1, dtype=tc.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = tc.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = tc.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: tc.Tensor):
        r""" Core forward function for SS2D.
        Performs SS on 4 sequence obtained by inspecting the input 2D feature map
        along different directions: (H→W), (W→H), (H→W flipped), (W→H flipped).
        """
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H*W # sequence length = N of tokens
        K = 4  # number of projections

        x_hwwh = tc.stack([x.view(B, -1, L), tc.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = tc.cat([x_hwwh, tc.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_bdl = tc.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = tc.split(x_bdl, [self.dt_rank, self.d_state, self.d_state], dim=2)  
        dts = tc.einsum("b k r l, k d r -> b k d l", dts.view(B,K,-1,L), self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1)
        As = -tc.exp(self.A_logs.float()).view(-1, self.d_state) 
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == tc.float

        inv_y = tc.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y =  tc.transpose(out_y[:, 1].view(B, -1, H, W), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = tc.transpose(inv_y[:, 1].view(B, -1, H, W), dim0=2, dim1=3).contiguous().view(B, -1, L)        
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = tc.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y
    
    def forward(self, x: tc.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # each: (B, H, W, d_inner)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0.0,
            d_state = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            **kwargs,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input: tc.Tensor):
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.drop_path(x)
        x = x + input
        return x
    

class VSSLayer(nn.Module):
    """ A basic VSS block (VMamba) for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            downsample=None, # downsample performe in patch merging
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        if downsample is not None and upsample is not None:
            raise ValueError("VSSLayer: choose either downsample or upsample, not both.")

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                **kwargs,
            ) for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        if upsample is not None:
            self.upsample = PatchExpand(dim, dim_scale=2, norm_layer=nn.LayerNorm)
        else:
            self.upsample = None
        
    def forward(self, x: tc.Tensor):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        if self.upsample is not None:
            x = self.upsample(x)
        
        return x

class MambaUNet(nn.Module):
    r""" Mamba U-Net
    A U-Net architecture based on Mamba SSM layers.
    """
    def __init__(   self,
                    patch_size=4,
                    in_chans=3,
                    num_classes=4,
                    depths=[2,2,6,2],
                    dims=[96, 192, 384, 768],
                    d_state=16,
                    drop_rate=0,
                    attn_drop_rate=0., 
                    drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm,
                    patch_norm=True,
                    use_checkpoint=False,
                    final_upsample="expand_first",
                    **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.d_state = math.ceil(dims[0] / 6) if d_state is None else d_state
        if isinstance(dims,int):
            dims = [int(dims * 2**i) for i in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.final_upsample = final_upsample

        self.patch_embed = PatchEmbed2D(patch_size=patch_size,
                                        in_chans=in_chans,
                                        embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        dpr = [x.item() for x in tc.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
    
        # Build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim = self.dims[i_layer],
                depth=depths[i_layer],
                d_state=self.d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,                         
            )
            self.layers.append(layer)
        
        # Build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(
                2*int(self.embed_dim*2**(self.num_layers-1-i_layer)),
                int(self.embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    dim=int(self.embed_dim*2**(self.num_layers-1-i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer,
                )
            else:
                layer_up = VSSLayer(
                    dim=int(self.embed_dim*2**(self.num_layers-1-i_layer)),
                    depth=depths[(self.num_layers-1-i_layer)],
                    d_state=self.d_state,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers -1) else None,
                    use_checkpoint=use_checkpoint,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(dim_scale=4,dim=self.embed_dim)
            self.output = nn.Conv2d(in_channels=self.embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    # Encoder and Bottleneck
    def forward_features(self, x: tc.Tensor):
        x = self.patch_embed(x)

        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        x = self.norm(x)
        return x, x_downsample

    # Decoder and Skip Connections
    def forward_up_features(self, x: tc.Tensor, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = tc.cat([x, x_downsample[self.num_layers - inx - 1]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        
        x = self.norm_up(x)
        return x
    
    def up_x4(self, x):
        if self.final_upsample=="expand_first":
            B,H,W,C = x.shape
            x = self.up(x)
            x = x.view(B, 4*H, 4*W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)
        return x

    def forward(self, x: tc.Tensor):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        return x
    
    def flops(self, shape=(3, 224, 224)):
        # to be implemented
        return
        

    



if __name__ == "__main__":
    B,N,H,W = 16,3,224,224
    B,N,H,W = 16,2,21,21
    patch_size = 3
    stride = 3
    x = tc.randn(B,N,H,W).cuda()
    # model = PatchEmbed2D(img_size=H, patch_size=patch_size, stride=stride, in_chans=N, embed_dim=96).cuda()
    model = MambaUNet(img_size=H, patch_size=patch_size, in_chans=N, embed_dim=96, stride=stride).cuda()
    out = model(x)
    print(out.shape)
    print("done1")   




