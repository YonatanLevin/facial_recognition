from torch.nn import Linear
import torch.nn.functional as F

from comparison_heads.comparison_head import ComparisonHead


class CosineHead(ComparisonHead):
    def __init__(self, encoding_dim: int):
        super().__init__(encoding_dim)

        self.ln = Linear(1, 1)

    def forward(self, img1, img2):
        cos_sim = F.cosine_similarity(img1, img2, dim=1)
        cos_sim = cos_sim.unsqueeze(1)
        logits = self.ln(cos_sim)
        return logits
    
    def logits_to_probs(self, logits):
        return F.sigmoid(logits)
        