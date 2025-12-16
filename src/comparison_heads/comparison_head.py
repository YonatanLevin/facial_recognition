from abc import ABC
from typing import Any

from torch import Tensor
from torch.nn import Module

class ComparisonHead(Module, ABC):
    def __init__(self, hyper_parameters: dict[str, Any]):
        super().__init__()
        self.hyper_parameters = hyper_parameters

    def forward(self, img1: Tensor, img2: Tensor, is_probs: bool) -> Tensor:
        """
        Calculate similarity score given two img embeddings
        
        :param img1: The embedding of the first image
        :type img1: Tensor
        :param img2: The embedding of the second image
        :type img2: Tensor
        :param img2: Whether to compute probabilities or logits
        :type img2: bool
        :return: Logits / similarity score
        :rtype: Tensor
        """
        pass