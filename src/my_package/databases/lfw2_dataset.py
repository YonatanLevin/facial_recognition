from os.path import join
from typing import Optional
import os

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm


from my_package.constants import DATA_DIR
from my_package.databases.img_transformer import ImgTransformer


class LFW2Dataset(Dataset):
    def __init__(self, 
                 is_train: bool, 
                 img_transformer: Optional[ImgTransformer] = None, 
                 resize_size: tuple[int, int] | None = None,
                 use_foreground: bool | None = False):
        """
        Args:
            is_train (bool): Whether this is the training set.
            img_transformer (Optional[ImgTransformer]): Standard augmentations (e.g. flips/rotations).
            resize_size (tuple[int, int] | None): Target size for resizing.
            use_foreground (bool): If True AND is_train is True, use images from the 
                                       'foreground' folder and apply random backgrounds.
            add_noise (bool): If True AND use_foreground is True, add Gaussian noise.
        """
        super().__init__()
        self.data_main_path = DATA_DIR
        self.is_train = is_train
        self.img_transformer = img_transformer
        self.resize_size = resize_size

        self.img_mean: Optional[Tensor] = None
        self.img_std: Optional[Tensor] = None

        self._cache = {} # Dictionary to store {path: Tensor}
        
        # Augmentation settings specifically for training with foregrounds
        self.use_foreground = use_foreground and is_train
        

        self.find_imgs_path_extension()
            
        self.pairs_df = self.parse_pairs()


    def find_imgs_path_extension(self) -> tuple[str, str]:
        # Determine correct path and extension based on augmentation setting
        if self.use_foreground:
            self.imgs_main_path = join(self.data_main_path, 'foreground')
            self.img_extension = '.png' # Foregrounds hold alpha channel
        elif self.resize_size[0] == 105 and self.resize_size[1] == 105:
            self.imgs_main_path = join(self.data_main_path, 'resized')
            self.img_extension = '.jpg'
        else:
            self.imgs_main_path = join(self.data_main_path, 'lfw2')
            self.img_extension = '.jpg'
        
    def parse_pairs(self) -> pd.DataFrame:
        """
        Parse the labels pairs to a dataframe.
        colums = [name1, idx1, name2, idx2]
        """
        pairs_file_name = 'pairsDevTrain.txt' if self.is_train else 'pairsDevTest.txt'
        pairs_file_path = join(self.data_main_path, pairs_file_name)
        rows_data = []
        try:
            with open(pairs_file_path, 'r', encoding='utf-8') as file:
                next(file) # skip the {rows_count} leading row
                for line in file:
                    line_data = line.split()
                    if len(line_data) == 3:
                        name1, idx1, idx2 = line_data
                        name2 = name1
                    elif len(line_data) == 4:
                        name1, idx1, name2, idx2 = line_data
                    else:
                        raise RuntimeError(f'Unrecognized pair row form:\n {line}')
                    rows_data.append({'name1': name1, 'idx1': idx1, 'name2': name2, 'idx2': idx2})
        except FileNotFoundError:
             print(f"Warning: Pairs file not found at {pairs_file_path}. Create dummy data for testing.")
             # Create dummy data so the class compiles if files are missing
             return pd.DataFrame(columns=['name1', 'idx1', 'name2', 'idx2'])

        return pd.DataFrame(rows_data)

    def __len__(self):
        return self.pairs_df.shape[0]
    
    def constract_img_path(self, human_name: str, img_idx: str | int) -> str:
        """
        Construct an img's path using the correctly determined extension.
        """
        img_idx = int(img_idx)
        # Use self.img_extension instead of hardcoded .jpg
        img_name = human_name + f'_{img_idx:04d}{self.img_extension}'
        return join(self.imgs_main_path, human_name, img_name)

    def load_robust_image(self, path: str) -> Tensor:
        """
        Load an image with PIL, handles resizing, random background augmentation, 
        and noise injection if configured for training. Defaults to grayscale tensor.
        """
        with Image.open(path) as img:
            img = img.convert('L')
            if self.resize_size is not None and not (self.resize_size[0] == 105 and self.resize_size[1] == 105):
                
                img = img.resize(self.resize_size, Image.Resampling.LANCZOS)
                
        return img

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, bool]:
        name1, idx1, name2, idx2 = self.pairs_df.iloc[index]
        label = name1 == name2
        
        img1_path = self.constract_img_path(name1, idx1)
        img2_path = self.constract_img_path(name2, idx2)

        # Load images with potential background/noise augmentation
        img1 = self.get_image(img1_path)
        img2 = self.get_image(img2_path)

        img1 = to_tensor(img1)
        img2 = to_tensor(img2)

        if self.img_mean is not None:
            img1 = (img1 - self.img_mean) / self.img_std
            img2 = (img2 - self.img_mean) / self.img_std

        # Apply standard geometric augmentations (flips, rotations) afterward
        if self.img_transformer:
            img1 = self.img_transformer(img1)
            img2 = self.img_transformer(img2)

        return img1, img2, label

    def get_image(self, path: str) -> Tensor:
        """Helper to manage the cache lookup and storage."""
        if path not in self._cache:
            self._cache[path] = self.load_robust_image(path)
            
        return self._cache[path]
    
    def calc_images_mean_std(self, indices: list[int] | None = None) -> tuple[Tensor, Tensor]:
        """
        Calculates mean and std. 
        If indices are provided, only uses those samples (important for train/val split).
        """
        # Determine which rows to process
        subset_df = self.pairs_df.iloc[indices] if indices is not None else self.pairs_df
        
        unique_paths = set()
        for _, row in subset_df.iterrows():
            unique_paths.add(self.constract_img_path(row['name1'], row['idx1']))
            unique_paths.add(self.constract_img_path(row['name2'], row['idx2']))

        if not unique_paths:
            return torch.zeros(1), torch.ones(1)

        all_images = []
        for path in tqdm(unique_paths, desc="Calculating Mean/Std"):
            img = self.get_image(path)
            all_images.append(to_tensor(img))

        stacked = torch.stack(all_images, dim=0) # [N, 1, H, W]
        mean = torch.mean(stacked, dim=0)
        std = torch.std(stacked, dim=0) + 1e-6 # Add epsilon to avoid div by zero
        return mean, std
    
    def set_normalization_stats(self, mean: Tensor, std: Tensor):
        """Method to inject mean and std calculated from training data."""
        self.img_mean = mean
        self.img_std = std
    