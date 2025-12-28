import torch
from torch.optim.lr_scheduler import StepLR

from my_package.model import Model
from my_package.learners.learner import Learner
from my_package.siamese_encoders.paper_cnn import PaperCNN
from my_package.comparison_heads.paper_head import PaperHead


class PaperLearner1(Learner):
    def __init__(self, device, resize_size: tuple[int, int] | None = None, 
                 use_foreground: bool=False):
        encoder = PaperCNN()
        head = PaperHead(encoder.encoding_dim)
        super().__init__(Model(encoder, head), device, resize_size=resize_size, 
                         use_foreground=use_foreground)
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

    def setup_loss(self, train_positive_percent):
        pos_weight_val = (1 - train_positive_percent) / train_positive_percent    
        pos_weight_tensor = torch.tensor([pos_weight_val])
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)