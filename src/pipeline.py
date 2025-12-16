import json
from os.path import join
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from databases.lfw2_dataset import LFW2Dataset
from model import Model
from comparison_heads.comparison_head import ComparisonHead
from siamese_encoders.encoder import Encoder
from siamese_encoders.paper_cnn import PaperCNN
from comparison_heads.paper_head import PaperHead

class Pipeline():
    def __init__(self):
        self.config_dir = 'model_configs'
        self.train_loader, self.val_loader, self.test_loader = setup_loaders(val_ratio=0.2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)

        self.max_epochs = 30
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        config_dict = self.load_config('conf1')
        self.model = create_model(config_dict)
        self.optimizer = torch.optim.SGD(self.model.parameters())
        self.model.to(self.device)

        self.train()

    def train(self):
        for epoch in range(self.max_epochs):
            self.train_epoch(epoch)

    def train_epoch(self, epoch: int):
        self.model.train()
        for img1, img2, label in tqdm(self.train_loader, desc=f'Training. Epoch: {epoch}'):
            img1, img2 = img1.to(self.device), img2.to(self.device)
            label = label.to(self.device).float().unsqueeze(1)

            prediction = self.model(img1, img2, False)
            loss = self.loss_fn(prediction, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
    return Model(encoder, head)
        
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

def setup_loaders(val_ratio: float) -> tuple[DataLoader, DataLoader, DataLoader]:
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

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, val_loader, test_loader