from os.path import join
from typing import Optional

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image

from databases.img_transformer import ImgTransformer

class LFW2Dataset(Dataset):
    def __init__(self, is_train: bool, img_transformer: Optional[ImgTransformer] = None, 
                 resize_size: tuple[int, int] | None = None):
        super().__init__()
        self.data_main_path = 'data'
        self.imgs_main_path = join(self.data_main_path, 'lfw2')
        self.is_train = is_train
        self.img_transformer = img_transformer
        self.resize_size = resize_size
        
        self.pairs_df = self.parse_pairs()
        
    def parse_pairs(self) -> pd.DataFrame:
        """
        Parse the labels pairs to a dataframe.
        colums = [name1, idx1, name2, idx2]
        """
        pairs_file_name = 'pairsDevTrain.txt' if self.is_train else 'pairsDevTest.txt'
        pairs_file_path = join(self.data_main_path, pairs_file_name)
        rows_data = []
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
        return pd.DataFrame(rows_data)

    def __len__(self):
        return self.pairs_df.shape[0]
    
    def constract_img_path(self, human_name: str, img_idx: str | int) -> str:
        """
        Constract an img's path from the humans name and the img index
        
        :param human_name: The name of the human
        :type human_name: str
        :param img_idx: The index of the image
        :type img_idx: str | int
        :return: The relative path of the img in the project
        :rtype: str
        """
        img_idx = int(img_idx)
        img_name = human_name + f'_{img_idx:04d}.jpg'
        return join(self.imgs_main_path, human_name, img_name)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, bool]:
        """
        Get an item from the dataset by index
        :param index: The index of the pair in the folder
        :type index: int
        :return: The two images and their pair label
        :rtype: tuple[Tensor, Tensor, bool]
        """
        name1, idx1, name2, idx2 = self.pairs_df.iloc[index]
        label = name1 == name2
        
        img1_path = self.constract_img_path(name1, idx1)
        img2_path = self.constract_img_path(name2, idx2)

        img1 = self.load_robust_image(img1_path)
        img2 = self.load_robust_image(img2_path)

        if self.img_transformer:
            img1 = self.img_transformer(img1)
            img2 = self.img_transformer(img2)

        return img1, img2, label
    

    def load_robust_image(self, path: str) -> Tensor:
        """
        Load an image with PIL
        
        :param path: Image path
        :type path: str
        :return: Loaded image tensor
        :rtype: Tensor
        """
        with Image.open(path) as img:
            img = img.convert('L')
            if self.resize_size is not None:
               img = img.resize(self.resize_size, Image.Resampling.LANCZOS)
            return to_tensor(img)
        