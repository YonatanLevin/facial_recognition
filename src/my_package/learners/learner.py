from abc import ABC

import torch
from torch import Tensor

from my_package.model import Model

class Learner(ABC):
    def __init__(self, model: Model, device: torch.device, resize_size: int | None = None, 
                 use_foreground: bool = False, normalize_imgs: bool = False):
        super().__init__()
        self.model = model
        self.device = device
        self.resize_size = resize_size
        self.use_foreground = use_foreground
        self.normalize_imgs = normalize_imgs

        self.model.to(self.device)

        print('model parameters number: ', sum(p.numel() for p in self.model.parameters()))
        
    def process_batch(self, img1: Tensor, img2: Tensor, label: Tensor, is_train: bool) -> tuple[Tensor, Tensor]:
        """
        Process a batch. Optionaly make an optimization step
        :param img1: The first images
        :type img1: Tensor
        :param img2: The second images
        :type img2: Tensor
        :param label: The pairs' labels
        :type label: Tensor
        :param is_train: Whether to make an optimization step
        :type is_train: bool
        :return: probs and loss
        :rtype: tuple[Tensor, Tensor]
        """
        pass

    def finish_epoch(self):
        """
        Respond to epoch end event.
        """
        pass

    def set_train(self, mode: bool):
        self.model.train(mode)

    def setup_loss(self, train_positive_percent: float):
        pass