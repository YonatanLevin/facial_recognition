import torch
from torch.optim.lr_scheduler import StepLR

from model import Model
from learners.learner import Learner
from siamese_encoders.paper_cnn import PaperCNN
from comparison_heads.paper_head import PaperHead


class PaperLearner(Learner):
    def __init__(self, device):
        encoder = PaperCNN()
        head = PaperHead(encoder.encoding_dim)
        super().__init__(Model(encoder, head), device)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters())
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.99)

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