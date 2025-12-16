import torch
from torch.utils.data import DataLoader, random_split

from databases.lfw2_dataset import LFW2Dataset
from model import Model

class Pipeline():
    def __init__(self):
        self.train_loader, self.val_loader, self.test_loader = self.setup_loaders(val_ration=0.2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.max_epochs = 30
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def setup_loaders(val_ration: float) -> tuple[DataLoader, DataLoader, DataLoader]:
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
        val_ratio = 0.2
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
         
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_dataset, val_loader, test_loader

    def train(self, model: Model):
        optimizer = torch.optim.SGD(model.parameters())
        for epoch in range(self.max_epochs):
            self.train_epoch(model, optimizer)

    def train_epoch(self, model: Model, optimizer: torch.optim.Optimizer):
        self.model.train()
        for img1, img2, label in self.train_loader:
            img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
            
            prediction = self.model(img1, img2)
            loss = self.loss_fn(prediction, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def optimize_model(model: Model):
        pass

    def optimize_models():
        pass