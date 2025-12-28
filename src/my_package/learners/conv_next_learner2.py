import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from my_package.model import Model
from my_package.learners.learner import Learner
from my_package.comparison_heads.paper_head import PaperHead
from my_package.siamese_encoders.conv_next import ConvNeXt


class ConvNeXtLearner2(Learner):
    def __init__(self, device, resize_size: tuple[int, int] | None = (105,105), 
                 use_foreground: bool=False):
        encoder = ConvNeXt()
        head = PaperHead(encoder.encoding_dim)
        super().__init__(Model(encoder, head), device, resize_size=resize_size, 
                         use_foreground=use_foreground)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters())
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5)

    def process_batch(self, img1, img2, label, is_train):
        encoding1, encoding2, logits = self.model(img1, img2)
        probs = self.model.logits_to_probs(logits)
        loss = self.loss_fn(logits, label)

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return probs, loss

    def finish_epoch(self):
        self.scheduler.step()