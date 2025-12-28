import torch
from torch.optim.lr_scheduler import StepLR

from model import Model
from learners.learner import Learner
from comparison_heads.paper_head import PaperHead
from siamese_encoders.conv_next import ConvNeXt


class ConvNeXtLearner1(Learner):
    def __init__(self, device, resize_size: tuple[int, int] | None = (105,105), 
                 use_foreground: bool=False):
        encoder = ConvNeXt()
        head = PaperHead(encoder.encoding_dim)
        super().__init__(Model(encoder, head), device, resize_size=resize_size, 
                         use_foreground=use_foreground)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters())
        # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.99)

    def process_batch(self, img1, img2, label, is_train):
        encoding1, encoding2, logits = self.model(img1, img2)
        probs = self.model.logits_to_probs(logits)
        loss = self.loss_fn(logits, label)

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return probs, loss

    # def finish_epoch(self):
    #     self.scheduler.step()