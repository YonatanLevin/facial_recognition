import json
from os import makedirs, listdir
from os.path import join, isfile 
from typing import Any

import numpy as np
import networkx as nx
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from databases.lfw2_dataset import LFW2Dataset
from databases.img_transformer import ImgTransformer
from databases.affine_transformer import AffineTransformer
from learners.paper_learner import PaperLearner

class Pipeline():
    def __init__(self):
        self.device = find_device()
        print('device: ', self.device)

        self.max_epochs = 60
        self.val_ratio=0.2
        self.batch_size = 128
        self.num_workers = 4

        self.learners_dict = {'PaperLearner': PaperLearner}
        self.learner, self.learner_name = None, None

        self.history_path = 'history.csv'
        self.history_plots_dir = 'history_plots'
        self.history_df = self.load_history()

        self.train_loader, self.val_loader, self.test_loader = None, None, None
        self.img_transformer = None

    def execute_learner(self):
        """
        Execute a learner
        """
        self.learner_name = self.request_learner_name()
        learner_class = self.learners_dict.get(self.learner_name)

        if learner_class == None:
            return
        
        self.learner = learner_class(self.device)
        self.img_transformer = create_img_transformer()
        self.train_loader, self.val_loader, self.test_loader = self.setup_loaders()
        
        self.train_metrics, self.val_metrics = self.setup_metrics()
        
        self.train()

        # self.epoch(0, 'test')

    def train(self):
        for epoch in range(self.max_epochs):
            # Training Phase
            self.epoch(epoch, phase='train')
            # Validation Phase
            self.epoch(epoch, phase='eval')
            # Step scheduler after the full epoch cycle
            self.learner.finish_epoch()
            self.print_metrics()

        self.history_df = self.history_df[self.history_df['learner'] != self.learner_name]
        self.history_df.to_csv(self.history_path,index=False)

    def epoch(self, epoch: int, phase: str):
        is_train = phase == 'train'
        self.learner.set_train(is_train)
        data_loader = self.train_loader if is_train else (self.val_loader if phase == 'eval' else self.test_loader)
        
        # Select the appropriate metric tracker
        current_metrics = self.train_metrics if is_train else self.val_metrics
        current_metrics.reset()

        running_loss = torch.tensor(0.0).to(self.device)

        with torch.set_grad_enabled(is_train):
            for img1, img2, label in tqdm(data_loader, desc=f'{phase}. Epoch: {epoch}'):
                img1, img2 = img1.to(self.device, non_blocking=True), img2.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True, dtype=torch.float32).unsqueeze(1)

                probs, loss = self.learner.process_batch(img1, img2, label, is_train)
                # Update metrics on GPU
                running_loss += loss
                current_metrics.update(probs, label)

        # Compute final metrics for the epoch
        computed_metrics = current_metrics.compute()
        
        avg_loss = running_loss.item() / len(data_loader)
        
        # Prepare metrics for history_df (move to CPU only at the very end)
        epoch_results = {
            'learner': self.learner_name,
            'phase': phase,
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': computed_metrics[f'{phase}_accuracy'].item(),
            'precision': computed_metrics[f'{phase}_precision'].item(),
            'recall': computed_metrics[f'{phase}_recall'].item(),
            'f1': computed_metrics[f'{phase}_f1'].item()
        }
        
        self.history_df.loc[len(self.history_df)] = epoch_results
        print(epoch_results)

    def load_history(self):
        if isfile(self.history_path):
            history_df = pd.read_csv(self.history_path, index_col=None)
            return history_df
        return pd.DataFrame(columns=['learner', 'phase', 'epoch', 'loss', 'accuracy', 'precision', 'recall', 'f1'])

    def setup_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create identity-disjoint loaders using Graph Connected Components.
        Splits are performed greedily to reach the target validation ratio.
        """
        # 1. Initialize original datasets
        resize_size = self.learner.resize_size
        use_foreground = self.learner.use_foreground
        test_dataset = LFW2Dataset(is_train=False, resize_size=resize_size, use_foreground=use_foreground)
        
        train_dataset, val_dataset = self.split_train_val(resize_size, use_foreground=use_foreground)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader
    
    def split_train_val(self, resize_size: tuple[int,int], use_foreground: bool | None) -> tuple[Subset, Subset]:
        """
        Load and split the full train dataset to train and val datasets.
        Split connected components based on val_ratio probability.

        :param resize_size: The resize_size
        :type resize_size: tuple[int, int]
        :return: (train_dataset, test_dataset)
        :rtype: tuple[Subset, Subset]
        """
        full_train_base = LFW2Dataset(is_train=True, img_transformer=self.img_transformer, 
                                      resize_size=resize_size, use_foreground=use_foreground)
        full_val_base = LFW2Dataset(is_train=True, img_transformer=None, resize_size=resize_size, 
                                    use_foreground=use_foreground)

        df = full_train_base.pairs_df
        component_indices = calculate_data_connected_components(df)
        
        val_idx = []
        train_idx = []
        # Assign components based on val_ratio probability
        np.random.seed(42)
        for indices in component_indices:
            if np.random.rand() < self.val_ratio:
                val_idx.extend(indices)
            else:
                train_idx.extend(indices)

        # Reporting Statistics
        def print_stats(indices, name):
            if not indices:
                print(f"{name} Set: Empty")
                return
            subset_df = df.iloc[indices]
            pos = (subset_df['name1'] == subset_df['name2']).sum()
            neg = len(subset_df) - pos
            print(f"{name} Set: Positive={pos} ({pos/len(subset_df):.3f}%), Negative={neg}, Total={len(subset_df)}")

        print_stats(train_idx, "Train")
        print_stats(val_idx, "Validation")

        # 6. Create Subsets
        train_dataset = Subset(full_train_base, train_idx)
        val_dataset = Subset(full_val_base, val_idx)
        
        return train_dataset, val_dataset

    def setup_metrics(self) -> tuple[MetricCollection, MetricCollection]:
        """
        Setup train and val metrics
        
        :return: Metric collections
        :rtype: tuple[MetricCollection, MetricCollection]
        """
        metrics = MetricCollection({
            'accuracy': Accuracy(task='binary'),
            'precision': Precision(task='binary'),
            'recall': Recall(task='binary'),
            'f1': F1Score(task='binary')
        })
        self.train_metrics = metrics.clone(prefix='train_').to(self.device)
        self.val_metrics = metrics.clone(prefix='eval_').to(self.device)

        return self.train_metrics, self.val_metrics

    def print_metrics(self):
        learner_history_df = self.history_df[self.history_df['learner'] == self.learner_name]\
                                           .sort_values(by='epoch', ascending=True)
        learner_history_df = learner_history_df[learner_history_df['phase'] != 'test']
        metrics_cols = ['precision', 'accuracy', 'recall', 'f1']
        long_df = learner_history_df.melt(
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
        axes[0].set_title('Performance Metrics')
        axes[0].set_ylabel('Score')

        axes[0].set_ylim(bottom=0, top=1)

        sns.lineplot(data=learner_history_df, x='epoch', y='loss', hue='phase', ax=axes[1])
        axes[1].set_ylim(bottom=0)

        makedirs(self.history_plots_dir, exist_ok=True)
        plt.savefig(join(self.history_plots_dir, self.learner_name+'.png'))
        plt.close()

    def print_best_lerner_metrics(self):
        eval_df = self.history_df.query("phase == 'eval'")
        best_f1_per_learner = eval_df.loc[eval_df.groupby('learner')['f1'].idxmax()]
        print(best_f1_per_learner)

    def request_learner_name(self) -> str:
        print('Learners: ', list(self.learners_dict.keys()))
        return input('Learner name:')
        
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

def create_img_transformer() -> ImgTransformer:
    return AffineTransformer()

def calculate_data_connected_components(df: pd.DataFrame) -> list[list[int]]:
    """
    Calculate the person connected components in a dataframe
    
    :param df: The dataframe
    :type df: pd.DataFrame
    :return: The connected componets' indexes
    :rtype: list[list[int]]
    """
    G = nx.Graph()
    for idx, row in df.iterrows():
        # Add edge between name1 and name2; store index to retrieve later
        G.add_edge(row['name1'], row['name2'])
        if 'indices' not in G[row['name1']][row['name2']]:
            G[row['name1']][row['name2']]['indices'] = []
        G[row['name1']][row['name2']]['indices'].append(idx)

    # Find Connected Components and map back to DataFrame indices
    components = list(nx.connected_components(G))
    component_indices = []
    
    for comp in components:
        # Get all row indices where both name1 and name2 are in this component
        indices = df[df['name1'].isin(comp) & df['name2'].isin(comp)].index.tolist()
        component_indices.append(indices)

    # Sort components by the number of samples they contain
    component_indices.sort(key=len, reverse=False)
    return component_indices