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

class PatchEmbedVideo(nn.Module):
    """ Video to Patch Embedding
    Args:
        img_size (int): Frame size.  Default: 21
        in_chans (int): Number of input image channels. Default: 3
        n_frames (int): Number of frames. Default: 8
        patch_size (int): Patch token size. Default: 3
        stride (int): Stride of the patch embedding. Default: None
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, image_size=21, n_frames=8, patch_size=3, 
                 stride=None, in_chans=2, embed_dim=96, groups=1, 
                 norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if stride is None:
            stride = patch_size
        if isinstance(stride, int):
            stride = (stride, stride)

        self.proj = nn.Conv2d(in_chans, 
                              embed_dim, 
                              kernel_size=patch_size,
                              stride=patch_size,
                              groups=groups)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None 
        
    def forward(self, x: tc.Tensor):
        """ Forward function.
        Args:
            x: (tc.Tensor) Input video of shape (B, T, C, H, W)
        Returns:
            tc.Tensor: Patch embedded video of shape (B, H', W', embed_dim)
        """
        B, T, C, H, W = x.shape
         
        for t in range(T):
            x_t = x[:, t, :, :, :]  # (B, C, H, W)
            x_t = self.proj(x_t)  # (B, embed_dim, H', W')
            if t == 0:
                x_out = x_t.unsqueeze(1)  # (B, 1, embed_dim, H', W')
            else:
                x_out = tc.cat((x_out, x_t.unsqueeze(1)), dim=1)  # (B, T, embed_dim, H', W')
        

        if self.norm is not None:
            x = self.norm(x)
        
        return x