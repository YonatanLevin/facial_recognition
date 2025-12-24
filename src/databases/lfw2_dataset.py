from os.path import join
from typing import Optional
import os

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

from databases.img_transformer import ImgTransformer


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
        self.data_main_path = 'data'
        self.is_train = is_train
        self.img_transformer = img_transformer
        self.resize_size = resize_size
        
        # Augmentation settings specifically for training with foregrounds
        self.use_foreground = use_foreground and is_train

        # Determine correct path and extension based on augmentation setting
        if self.use_foreground:
            self.imgs_main_path = join(self.data_main_path, 'foreground')
            self.img_extension = '.png' # Foregrounds hold alpha channel
             # Safety check: ensure foreground directory exists
            if not os.path.exists(self.imgs_main_path):
                 raise FileNotFoundError(f"Foreground directory not found: {self.imgs_main_path}. Please run background removal first.")
        else:
            self.imgs_main_path = join(self.data_main_path, 'lfw2')
            self.img_extension = '.jpg'
            
        self.pairs_df = self.parse_pairs()
        
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
    
    def _add_random_background(self, rgba_img: Image.Image) -> Image.Image:
        """
        Takes an RGBA image, generates random RGB noise, and composites the
        foreground onto the noisy background using the alpha channel as a mask.
        """
        # 1. Generate random noise background
        # Create numpy array of random integers [0, 255] with RGB shape (H, W, 3)
        bg_np = np.random.randint(0, 256, (rgba_img.size[1], rgba_img.size[0], 3), dtype=np.uint8)
        bg_img = Image.fromarray(bg_np, 'RGB')

        # 2. Paste foreground onto background using alpha channel as mask
        # split()[3] is the alpha channel
        bg_img.paste(rgba_img, mask=rgba_img.split()[3])
        return bg_img

    def _add_gaussian_noise(self, rgb_img: Image.Image, noise_level: float = 15.0) -> Image.Image:
        """Optionally adds Gaussian noise to the final RGB image."""
        img_np = np.array(rgb_img).astype(np.float32)
        
        # Generate noise centered at 0 with standard deviation = noise_level
        noise = np.random.normal(loc=0.0, scale=noise_level, size=img_np.shape)
        
        noisy_img_np = img_np + noise
        
        # Clip values to valid [0, 255] range and convert back to uint8
        noisy_img_np = np.clip(noisy_img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img_np, 'RGB')

    def load_robust_image(self, path: str) -> Tensor:
        """
        Load an image with PIL, handles resizing, random background augmentation, 
        and noise injection if configured for training. Defaults to grayscale tensor.
        """
        try:
            with Image.open(path) as img:
                # 1. Resize first (standardize dimensions before processing)
                if self.resize_size is not None:
                    img = img.resize(self.resize_size, Image.Resampling.LANCZOS)

                # 2. Apply Foreground Augmentations (Only if training and configured)
                # We check img.mode == 'RGBA' to ensure it's a foreground image with transparency
                if self.use_foreground and img.mode == 'RGBA':
                    
                    # A. Replace transparent background with random noise
                    rgb_img = self._add_random_background(img)
                    
                    # B. Sdd extra Gaussian noise over the whole image
                    rgb_img = self._add_gaussian_noise(rgb_img)

                    # C. Convert to grayscale for final output
                    final_img = rgb_img.convert('L')
                    
                else:
                    # Standard loading for testing or normal training: just convert to grayscale
                    final_img = img.convert('L')

                # 3. Convert to tensor
                return to_tensor(final_img)
                
        except FileNotFoundError:
             print(f"Error: Image not found at {path}")
             # Return a black tensor of correct size as fallback to prevent crash
             if self.resize_size:
                 return torch.zeros((1, self.resize_size[1], self.resize_size[0]))
             raise

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, bool]:
        name1, idx1, name2, idx2 = self.pairs_df.iloc[index]
        label = name1 == name2
        
        img1_path = self.constract_img_path(name1, idx1)
        img2_path = self.constract_img_path(name2, idx2)

        # Load images with potential background/noise augmentation
        img1 = self.load_robust_image(img1_path)
        img2 = self.load_robust_image(img2_path)

        # Apply standard geometric augmentations (flips, rotations) afterward
        if self.img_transformer:
            img1 = self.img_transformer(img1)
            img2 = self.img_transformer(img2)

        return img1, img2, label