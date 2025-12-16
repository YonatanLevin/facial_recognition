from os.path import join

import seaborn
import pandas as pd
import numpy as np

from databases.lfw2_dataset import LFW2Dataset
from matplotlib import pyplot as plt


def main():
    plots_path = 'eda_plots/'
    train_dataset = LFW2Dataset(is_train=True)
    test_dataset = LFW2Dataset(is_train=False)

    plot_sample_pairs(train_dataset)

def plot_sample_pairs(dataset):
    n_pairs = 4
    fig, axes = plt.subplots(n_pairs,2)
    
    labels = []
    for i in range(n_pairs):
        img1, img2, label = dataset[i]
        img1 = img1[0,:,:].detach().cpu()
        img2 = img2[0,:,:].detach().cpu()
        axes[i,0].imshow(img1)
        axes[i,1].imshow(img2)

        axes[i,0].axis('off')
        axes[i,1].axis('off')

        labels.append(label)

    fig.tight_layout()
    
    plt.show()


if __name__ == '__main__':
    main()