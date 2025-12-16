from torch.nn import Module
from torch import Tensor

from siamese_encoders.encoder import Encoder
from comparison_heads.comparison_head import ComparisonHead

class Model(Module):
    def __init__(self, encoder: Encoder, head: ComparisonHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, img1: Tensor, img2: Tensor, is_probs: bool):
        img1_encoding = self.encoder(img1)
        img2_encoding = self.encoder(img2)
        y = self.head(img1_encoding, img2_encoding, is_probs)
        return y