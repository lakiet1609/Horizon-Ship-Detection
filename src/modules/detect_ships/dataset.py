import numpy as np
import cv2
import torch
from pathlib import Path

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, root='results/rotate', transform=False):
        super(Custom_Dataset, self).__init__()
        self.paths = list(Path(root).glob('*.png'))
        self.transform = transform
        self.mean = np.array([173, 148.5, 112.5], dtype=np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        img = cv2.imread(image_path)
        img = np.array(img, dtype=np.float32)
        img = (img - self.mean).transpose((2, 0, 1))
        return img