from abc import ABC
from typing import Any

from torch import Tensor
from torch.nn import Module

class ComparisonHead(Module, ABC):
    def __init__(self, encoding_dim: int):
        super().__init__()
        self.encoding_dim = encoding_dim

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        """
        Calculate similarity score given two img embeddings
        
        :param img1: The embedding of the first image
        :type img1: Tensor
        :param img2: The embedding of the second image
        :type img2: Tensor
        :return: Logits
        :rtype: Tensor
        """
        pass

    def logits_to_probs(self, logits: Tensor):
        pass