import torch
from torch.nn import Module
from torch import Tensor

from siamese_encoders.encoder import Encoder
from comparison_heads.comparison_head import ComparisonHead

class Model(Module):
    def __init__(self, encoder: Encoder, head: ComparisonHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, img1: Tensor, img2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        encoding1, encoding2 = self.encode(img1, img2)
        y = self.head(encoding1, encoding2)
        return encoding1, encoding2, y
    
    def encode(self, img1: Tensor, img2: Tensor) -> tuple[Tensor, Tensor]:
        combined = torch.cat([img1, img2], dim=0)
        encodings = self.encoder(combined)
        encoding1, encoding2 = torch.chunk(encodings, 2, dim=0)
        return encoding1, encoding2
    
    def logits_to_probs(self, logits: Tensor):
        return self.head.logits_to_probs(logits)