from abc import ABC

from torch import Tensor
from torch.nn import Module
from typing import Any


class Encoder(Module, ABC):
    """
    Encode images to an embedding space
    """
    def __init__(self, encoding_dim: int):
        super().__init__()
        self.encoding_dim = encoding_dim

    def forward(self, img: Tensor) -> Tensor:
        """
        Encode an img tensor to an embedding tensor
        
        :param img: The image tensor
        :type img: Tensor
        :return: an embedding tensor
        :rtype: Tensor
        """
        pass
