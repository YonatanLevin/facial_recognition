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
        encoding1 = self.encoder(img1)
        encoding2 = self.encoder(img2)
        y = self.head(encoding1, encoding2)
        return encoding1, encoding2, y
    
    def logits_to_probs(self, logits: Tensor):
        return self.head.logits_to_probs(logits)