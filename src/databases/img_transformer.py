from abc import ABC

from torch import Tensor

class ImgTransformer():
    def __init__(self):
        """
        Image transformer
        """
        super().__init__()

    def __call__(self, img: Tensor) -> Tensor:
        """
        Apply transformation on the image
        
        :param img: An image
        :type img: Tensor
        :return: The transformed image
        :rtype: Tensor
        """
        pass