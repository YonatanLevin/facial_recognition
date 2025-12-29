import torch
from my_package.model import Model
from my_package.learners.learner import Learner
from my_package.siamese_encoders.paper_cnn import PaperCNN
from my_package.comparison_heads.paper_head import PaperHead

class PaperLearner4(Learner):
    def __init__(self, device, resize_size: tuple[int, int] | None = (105, 105), 
                 use_foreground: bool=False):
        encoder = PaperCNN()
        head = PaperHead(encoder.encoding_dim)
        super().__init__(Model(encoder, head), device, resize_size=resize_size, 
                         use_foreground=use_foreground)
        
        # 1. Layer-wise Hyperparameters (from Bayesian Optimization / Paper specs)
        # The paper suggests these ranges: eta [1e-4, 1e-1], mu [0, 1], lambda [0, 0.1]
        # In a real scenario, these would be individual values per layer.
        self.lr_init = 1e-2 
        self.mu_final = 0.9  # Final individual momentum term mu_j
        self.weight_decay = 1e-2 # Layer-wise L2 regularization lambda_j
        
        # 2. Setup Optimizer with Layer-wise parameters
        # For simplicity here, we apply the same logic to all layers as the paper
        # allows for individual tuning, but uses a unified decay schedule.
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.lr_init, 
            momentum=0.5, # Initial momentum fixed to 0.5
            weight_decay=self.weight_decay
        )
        
        self.current_epoch = 0

    def process_batch(self, img1, img2, label, is_train):
        # The paper uses Binary Cross-Entropy (Regularized) [cite: 168, 169]
        encoding1, encoding2, logits = self.model(img1, img2)
        probs = self.model.logits_to_probs(logits)
        loss = self.loss_fn(logits, label)

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return probs, loss

    def finish_epoch(self):
        self.current_epoch += 1
        
        # 3. Learning Rate Decay: 1% per epoch 
        new_lr = self.lr_init * (0.99 ** self.current_epoch)
        
        # 4. Momentum Schedule: Linear increase until reaching mu_j 
        # Note: Standard PyTorch SGD doesn't support per-parameter momentum schedules easily 
        # without manual param_group updates.
        new_momentum = min(self.mu_final, 0.5 + (self.current_epoch * 0.05)) 

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            param_group['momentum'] = new_momentum

    def setup_loss(self, train_positive_percent):
        pos_weight_val = (1 - train_positive_percent) / train_positive_percent    
        pos_weight_tensor = torch.tensor([pos_weight_val]).to(self.device)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)