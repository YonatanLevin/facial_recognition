from typing import Any

import torch.nn.functional as F

from comparison_heads.comparison_head import ComparisonHead


class CosineHead(ComparisonHead):
    def __init__(self, hyper_parameters: dict[str, Any]):
        super.__init__(hyper_parameters)

    def forward(self, img1, img2):
        cos_sim = F.cosine_similarity(img1, img2, dim=1)
        return cos_sim
        