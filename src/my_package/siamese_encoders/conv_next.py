import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

from my_package.siamese_encoders.encoder import Encoder

class LayerNorm(nn.Module):
    """
    Standard LayerNorm that supports 'channels_first' (B, C, H, W) 
    by temporarily permuting to 'channels_last' (B, H, W, C).
    """
    def __init__(self, normalized_shape: int, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        # Move C to the end: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # Move C back: [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # 1. Depthwise Convolution (7x7)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        
        # 2. LayerNorm (Channel-wise)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # 3. Pointwise / Inverted Bottleneck (Linear layers are faster than 1x1 convs)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.dwconv(x)
        
        # Permute for LayerNorm and Linear Layers (Channels-Last)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Return to Channels-First
        x = x.permute(0, 3, 1, 2)
        
        return shortcut + x

class ConvNeXt(Encoder):
    def __init__(
        self, 
        encoding_dim: int = 384, 
        dims=[48, 96, 192, 384], 
        depths: List[int] = [2, 2, 6, 2]
    ):
        super().__init__(encoding_dim=encoding_dim)
        
        # 1. Stem: Patchify (105x105 -> 26x26)
        # Changed input channels to 1 for grayscale
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        # 2. Downsampling layers between stages
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)

        # 3. Feature extraction stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        # 4. Final Projection Head
        self.final_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.embedding_proj = nn.Linear(dims[-1], encoding_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Pass through downsampling and ConvNeXt stages
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # Global Average Pooling (B, C, H, W) -> (B, C)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        
        # Final Norm and Projection to 4096
        x = self.final_norm(x)
        embedding = self.embedding_proj(x)
        
        return embedding