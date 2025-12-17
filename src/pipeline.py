import json
from os.path import join
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Optimizer, SGD 
from torch.optim.lr_scheduler import LRScheduler, StepLR
from tqdm import tqdm

from databases.lfw2_dataset import LFW2Dataset
from databases.img_transformer import ImgTransformer
from databases.affine_transformer import AffineTransformer
from model import Model
from comparison_heads.comparison_head import ComparisonHead
from siamese_encoders.encoder import Encoder
from siamese_encoders.paper_cnn import PaperCNN
from comparison_heads.paper_head import PaperHead

class Pipeline():
    def __init__(self):
        self.config_dir = 'model_configs'
        self.device = find_device()
        print('device: ', self.device)

        self.max_epochs = 30
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        config_dict = self.load_config('conf1')
        img_transformer = create_img_transformer(config_dict)
        self.train_loader, self.val_loader, self.test_loader = setup_loaders(val_ratio=0.2, img_transformer=img_transformer)
        self.model = create_model(config_dict)
        self.model.to(self.device)
        self.optimizer = create_optimizer(config_dict, self.model)
        self.schedular = create_lr_scheduler(config_dict, self.optimizer)

        self.train()

    def train(self):
        for epoch in range(self.max_epochs):
            # Training Phase
            self.epoch(epoch, is_train=True)
            # Validation Phase
            self.epoch(epoch, is_train=False)
            # Step scheduler after the full epoch cycle
            self.scheduler.step()

    def epoch(self, epoch: int, is_train: bool):
        # 1. Setup mode and data source
        self.model.train() if is_train else self.model.eval()
        data_loader = self.train_loader if is_train else self.val_loader
        phase_name = "Training" if is_train else "Validation"
        
        running_loss = 0.0
        all_preds = []
        all_labels = []

        # 2. Enable/Disable gradient calculation
        with torch.set_grad_enabled(is_train):
            for img1, img2, label in tqdm(data_loader, desc=f'{phase_name}. Epoch: {epoch}'):
                img1, img2 = img1.to(self.device), img2.to(self.device)
                label = label.to(self.device, dtype=torch.float32).unsqueeze(1)

                # Forward pass
                prediction_logits = self.model(img1, img2, False)
                loss = self.loss_fn(prediction_logits, label)
                
                # 3. Optimization step (only during training)
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Statistics accumulation
                running_loss += loss.item()
                
                # Logic for metrics
                probs = torch.sigmoid(prediction_logits)
                preds = (probs > 0.5).float()
                
                all_preds.append(preds.detach().cpu())
                all_labels.append(label.detach().cpu())

        # 4. Final Metric Calculation
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # Calculate mean loss relative to the current loader length
        avg_loss = running_loss / len(data_loader)
        metrics = self.calculate_metrics(avg_loss, all_preds, all_labels)
        
        print(f"\n[{phase_name}] Epoch {epoch}:")
        print(f"Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.4f} | "
              f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

    def optimize_model(self, model: Model):
        pass

    def load_config(self, config_name: str | None = None):
        if config_name is None:
            config_name = input('config name:')
        
        config_file_path = join(self.config_dir, config_name + '.json')
        with open(config_file_path, 'r') as f:
            config_json = f.read()
        return json.loads(config_json)

def create_model(config_dict: dict[str, Any]) -> Model:
    encoder = create_encoder(config_dict)
    head = create_head(config_dict)
    return Model(encoder, head).float()
        
def find_device() -> torch.device:
    """
    Find apropriate computation hardware
    
    :return: Best device
    :rtype: device
    """
    device_name = 'cpu'
    if torch.cuda.is_available():
        device_name = 'cuda'
    elif torch.xpu.is_available():
        device_name = 'xpu'
    return torch.device(device_name)
    
def create_encoder(config_dict: dict[str, Any]) -> Encoder:
    """
    Create an encoder from the config_dict
    
    :param config_dict: The configuration dict
    :type config_dict: dict[str, Any]
    :return: Created encoder from config_dict
    :rtype: Encoder
    """
    encoder_name = config_dict['encoder_name']
    match encoder_name:
        case 'PaperCNN':
            return PaperCNN(config_dict)
        case _:
            raise ValueError(f'Unknown encoder: {encoder_name}')
        
def create_head(config_dict: dict[str, Any]) -> ComparisonHead:
    """
    Create a comparison head from the config dict
    
    :param config_dict: Config dict
    :type config_dict: dict[str, Any]
    :return: Comparison head from config_dict
    :rtype: ComparisonHead
    """
    head_name = config_dict['head_name']
    match config_dict['head_name']:
        case 'PaperHead':
            return PaperHead(config_dict)
        case _:
            raise ValueError(f'Unknown head: {head_name}')

def create_optimizer(config_dict: dict[str, Any], model: Model) -> Optimizer:
    return SGD(model.parameters())

def create_lr_scheduler(config_dict: dict[str, Any], optimizer: Optimizer) -> LRScheduler:
    return StepLR(optimizer, step_size=1, gamma=0.99) # Example: decay LR by gamma every step_size epochs

def create_img_transformer(config_dict: dict[str, Any]) -> ImgTransformer:
    return AffineTransformer()

def setup_loaders(val_ratio: float, img_transformer: ImgTransformer) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, val and test dataloaders
        
        :param val_ration: Description
        :type val_ration: float
        :return: Description
        :rtype: tuple[DataLoader, DataLoader, DataLoader]
        """
        train_dataset = LFW2Dataset(is_train=True)
        test_dataset = LFW2Dataset(is_train=False)

        total_size = len(train_dataset)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
         
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        train_dataset.img_transformer = img_transformer

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, val_loader, test_loader