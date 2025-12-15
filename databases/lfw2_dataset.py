from os import listdir, scandir
from os.path import join

from torch.utils.data import Dataset

class LFW2Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data_main_path = 'data\lfw2'
        self.human_names = [f.name for f in scandir(self.data_main_path)]
        self.img_names = {human_name: [f for f in listdir(join(self.data_main_path, human_name))] 
                     for human_name in self.human_names}

    def __len__(self):
        return 