
import torch

from my_package.comparison_heads.cosine_head import CosineHead
from my_package.learners.learner import Learner
from my_package.model import Model
from my_package.siamese_encoders.paper_cnn import PaperCNN


class CNNCosineLearner3(Learner):
    def __init__(self, device, use_foreground: bool=False, encoder_final_activation = None):
        encoder = PaperCNN(final_activation=encoder_final_activation)
        head = CosineHead(encoder.encoding_dim)
        super().__init__(Model(encoder, head), device, resize_size=(105, 105), 
                         use_foreground=use_foreground)
        self.cosine_loss_margin = 0.5
        self.embeding_loss = torch.nn.CosineEmbeddingLoss(self.cosine_loss_margin)
        self.head_loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.nn.CosineAnnealingWarmRestarts(self.optimizer, T_0=5)
        
    def process_batch(self, img1, img2, label, is_train):
        encoding1, encoding2 = self.model.encode(img1, img2)
        logits = self.model.head(encoding1, encoding2)
        probs = self.model.logits_to_probs(logits)

        # Calculate separate losses
        # loss_head will only update self.model.head parameters
        loss_head = self.head_loss(logits, label)
        
        # loss_embed will update self.model.encoder parameters
        cosine_labels = label.clone().squeeze()
        cosine_labels[cosine_labels == 0] = -1
        loss_embed = self.embeding_loss(encoding1, encoding2, cosine_labels)

        # Combine and Step
        total_loss = loss_head + loss_embed

        if is_train:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return probs, total_loss

    def finish_epoch(self):
        self.scheduler.step()
