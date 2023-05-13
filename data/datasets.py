import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PatchesDataset(Dataset): 
    def __init__(self, root_dataset, transforms=None):
        self.root = root_dataset
        self.data_df = pd.read_csv(os.path.join(self.root, 'label.csv'))
        self.image_folder = os.path.join(self.root, 'patches_samples')
        self.transforms = transforms

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        img1_filepath = os.path.join(self.image_folder, self.data_df.iloc[index, 0])
        img2_filepath = os.path.join(self.image_folder, self.data_df.iloc[index, 1])
        label = self.data_df.iloc[index, 2]
        img1 = Image.open(img1_filepath).convert('RGB')
        img2 = Image.open(img2_filepath).convert('RGB')

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return (img1, img2), torch.tensor(label, dtype=torch.float32)
