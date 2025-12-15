import seaborn
import pandas as pd

from databases.lfw2_dataset import LFW2Dataset
from matplotlib import pyplot as plt
from os.path import join

def main():
    plots_path = 'eda_plots/'
    dataset = LFW2Dataset()
    img_per_human_cdf(plots_path, dataset)

    
def img_per_human_cdf(plots_path: str, dataset: LFW2Dataset):
    """
    Plot the cdf of the number if images per human in the dataset.
    
    :param plots_path: Description
    :type plots_path: str
    :param dataset: Description
    :type dataset: LFW2Dataset
    """
    name_counts_df = pd.DataFrame({
    'name': list(dataset.img_names.keys()),
    'count': [len(img_names) for img_names in dataset.img_names.values()]
    })
    
    plt.figure(figsize=(10, 6)) # Explicitly create a new figure
    seaborn.ecdfplot(data=name_counts_df, x='count')

    # Add labels/title for clarity
    plt.title('Image Count Distribution')
    plt.xlabel('Number of Images per Person')
    plt.ylabel('Frequency')
    plt.xscale('log')

    plt.savefig(join(plots_path, 'human_img_count_cdf.png'))

    name_counts_df.sort_values(by='count',ascending=False,inplace=True)
    print(name_counts_df.head(3))

if __name__ == '__main__':
    main()