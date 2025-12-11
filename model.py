from torch.nn import Module
from torch import Tensor

class Model(Module):
    def __init__(self):
        super().__init__(Model, self)
        self.encoder = None
        self.head = None

    def forward(self, x: Tensor):
        x = self.encoder.forward(x)
        x = self.head.forward(x)
        return x