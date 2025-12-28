import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

from my_package.siamese_encoders.encoder import Encoder

# Helper for Stochastic Depth (Standard in modern CNNs/ViTs)
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0., expansion_ratio: float = 2.0):
        super().__init__()
        # 1. Depthwise Convolution (Reduced to 5x5 or kept at 7x7)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # 2. LayerNorm
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # 3. Inverted Bottleneck with lower expansion ratio
        hidden_dim = int(dim * expansion_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # Channels-Last
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # Channels-First
        
        return shortcut + self.drop_path(x)

class ConvNeXt(Encoder):
    def __init__(
        self, 
        encoding_dim: int = 512, # Standard for Face Verification
        dims: List[int] = [64, 128, 256, 512], # Reduced width
        depths: List[int] = [2, 2, 6, 2], # Reduced depth
        drop_path_rate: float = 0.1,
        expansion_ratio: float = 2.0
    ):
        # Pass 512 to the abstract Encoder class
        super().__init__(encoding_dim=encoding_dim)
        
        # Stochastic depth decay rule
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        # 1. Stem
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        # 2. Downsampling layers
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)

        # 3. Stages
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(
                    dim=dims[i], 
                    drop_path=dp_rates[cur + j], 
                    expansion_ratio=expansion_ratio
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # 4. Optimized Face Head
        self.final_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(p=0.3)
        self.embedding_proj = nn.Linear(dims[-1], encoding_dim)

    def forward(self, x: Tensor) -> Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        
        x = self.final_norm(x)
        x = self.dropout(x)
        embedding = self.embedding_proj(x)
        
        # Crucial for Cosine Similarity matching
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding