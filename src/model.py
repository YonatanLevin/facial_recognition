from torch.nn import Module
from torch import Tensor

from siamese_encoders.encoder import Encoder
from comparison_heads.comparison_head import ComparisonHead

class Model(Module):
    def __init__(self, encoder: Encoder, head: ComparisonHead):
        super().__init__(Model, self)
        self.encoder = encoder
        self.head = head

    def forward(self, img1: Tensor, img2: Tensor):
        img1_encoding = self.encoder.forward(img1)
        img2_encoding = self.encoder.forward(img2)
        y = self.head.forward(img1_encoding, img2_encoding)
        return y