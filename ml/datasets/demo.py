from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from pathlib import Path
from typing import List

class CatsAnDogs(Dataset):
    def __init__(self, images_paths: List[Path], transforms=None):
        self.images_paths = images_paths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        im_path = self.images_paths[idx]
        im = Image.open(im_path)
        if self.transforms:
            im = self.transforms(im)
        label = im_path.stem.split('.')[0]
        class_ = 0 if label == 'cat' else 1
        
        hot_label = np.eye(2)[class_]
        
        return im, torch.tensor(hot_label, dtype=torch.float32)