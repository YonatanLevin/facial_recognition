import json
from os.path import join, isfile, makedirs
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Optimizer, SGD 
from torch.optim.lr_scheduler import LRScheduler, StepLR
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import seaborn as sns

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

        self.max_epochs = 60
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.config_name = 'conf1'

        config_dict = self.load_config(self.config_name)

        self.history_path = 'history.csv'
        self.history_plots_dir = 'history_plots'
        self.history_df = self.load_history()

        img_transformer = create_img_transformer(config_dict)
        self.train_loader, self.val_loader, self.test_loader = setup_loaders(val_ratio=0.2, batch_size=128, 
                                                                             img_transformer=img_transformer)
        self.model = create_model(config_dict)
        self.model.to(self.device)
        self.optimizer = create_optimizer(config_dict, self.model)
        self.scheduler = create_lr_scheduler(config_dict, self.optimizer)

        self.train()
        self.epoch(0, 'test')

    def train(self):
        for epoch in range(self.max_epochs):
            # Training Phase
            self.epoch(epoch, phase='train')
            # Validation Phase
            self.epoch(epoch, phase='eval')
            # Step scheduler after the full epoch cycle
            self.scheduler.step()
            self.print_metrics()

    def epoch(self, epoch: int, phase: str):
        # 1. Setup mode and data source
        is_train = phase == 'train'
        if is_train:
            self.model.train()
            data_loader = self.train_loader
        else:
            self.model.eval()
            data_loader = self.val_loader if phase == 'eval' else self.test_loader
        
        running_loss = 0.0
        all_preds = []
        all_labels = []

        # 2. Enable/Disable gradient calculation
        with torch.set_grad_enabled(is_train):
            for img1, img2, label in tqdm(data_loader, desc=f'{phase}. Epoch: {epoch}'):
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
                probs = torch.sigmoid(prediction_logits).detach().cpu() 
                preds = (probs > 0.5).float()

                all_preds.append(preds)
                all_labels.append(label.detach().cpu())

        # 4. Final Metric Calculation
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # Calculate mean loss relative to the current loader length
        avg_loss = running_loss / len(data_loader)
        metrics = self.calculate_metrics(phase, epoch, avg_loss, all_preds, all_labels)
        
        self.history_df.loc[len(self.history_df)] = metrics
        print(metrics)

    def calculate_metrics(self, phase: str, epoch: int, loss: float, preds: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        """
        Calculates classification performance metrics.
        
        :param loss: The average loss for the epoch.
        :type loss: float
        :param preds: Predicted binary labels (0 or 1).
        :type preds: np.ndarray
        :param labels: Ground truth binary labels.
        :type labels: np.ndarray
        :return: A dictionary containing the calculated metrics.
        :rtype: dict
        """
        metrics = {
            'config': self.config_name,
            'phase': phase,
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0)
        }
        return metrics

    def optimize_model(self, model: Model):
        pass

    def load_config(self, config_name: str | None = None):
        if config_name is None:
            config_name = input('config name:')
        
        config_file_path = join(self.config_dir, config_name + '.json')
        with open(config_file_path, 'r') as f:
            config_json = f.read()
        return json.loads(config_json)

    def load_history(self):
        if isfile(self.history_path):
            return pd.read_csv(self.history_path)
        return pd.DataFrame(columns=['config', 'phase', 'epoch', 'loss', 'accuracy', 'precision', 'recall', 'f1'])

    def print_metrics(self):
        config_history_df = self.history_df[self.history_df['config'] == self.config_name]\
                                           .sort_values(by='epoch', ascending=True)
        config_history_df = config_history_df[config_history_df['phase'] != 'test']
        metrics_cols = ['precision', 'accuracy', 'recall', 'f1']
        long_df = config_history_df.melt(
            id_vars=['epoch', 'phase'], 
            value_vars=metrics_cols, 
            var_name='metric', 
            value_name='score'
        )

        fig, axes = plt.subplots(2, 1)
        fig.set_size_inches(10, 20)
        plot = sns.lineplot(
            data=long_df, 
            x='epoch', 
            y='score', 
            hue='metric',   # Different color for each metric
            style='phase',  # Different line style (solid/dashed) for train/val
            markers=True,   # Optional: add markers for better readability
            ax=axes[0]
        )

        # Refine the legend
        axes[0].legend(title='Metrics & Phase', loc='lower left')
        axes[0].set_title('Model Performance Metrics')
        axes[0].set_ylabel('Score')

        sns.lineplot(data=config_history_df, x='epoch', y='loss', hue='phase', ax=axes[1])

        makedirs(self.history_plots_dir, exist_ok=True)
        plt.savefig(join(self.history_plots_dir, self.config_name+'.png'))


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

def setup_loaders(val_ratio: float, batch_size: int, img_transformer: ImgTransformer) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders with identity-disjoint splits.
    """
    # 1. Initialize original datasets
    full_train_dataset = LFW2Dataset(is_train=True)
    test_dataset = LFW2Dataset(is_train=False)

    # 2. Extract names to create groups
    # Assuming full_train_dataset.df contains the columns 'name1' and 'name2'
    df = full_train_dataset.pairs_df
    
    # We define a 'group' for each row. To be strict, if a row contains 
    # Name A and Name B, that row belongs to a meta-group of those two names.
    # A simple approach is to use the primary name or a concatenated unique ID.
    groups = df['name1'] 

    # 3. Perform Group-wise Splitting
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    
    # split() yields indices for the train and val sets
    train_idx, val_idx = next(gss.split(X=df, y=None, groups=groups))

    # 4. Create Subsets
    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)

    # 5. Apply transformations
    # Note: Accessing the underlying dataset to set the transformer
    full_train_dataset.img_transformer = img_transformer

    # 6. Initialize Loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader