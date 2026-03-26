# dataset.py
import os
import numpy as np
import torch
import cv2
import tifffile as tiff
from config import Config
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, label=1):
        self.root_dir = root_dir
        self.transform = transform
        self.label = label
        self.images = [f for f in os.listdir(root_dir) if f.endswith("_C1.tiff")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root_dir, filename)
        try:
            image = tiff.imread(img_path)
            if len(image.shape) == 3:
                image = image[..., 0]
            image = image.astype(np.float32) / 65535.0  # Normalize 16-bit TIFF
            image = cv2.resize(image, Config.IMAGE_SIZE)
            image = np.expand_dims(image, axis=-1)
            if self.transform:
                image = self.transform(image)
            return image.to(torch.float32), torch.tensor([self.label], dtype=torch.float32), filename
        except Exception as e:
            print(f"(Error: {e})")
            # Return a dummy sample along with its filename so that the worker doesn't crash
            dummy = np.zeros((Config.IMAGE_SIZE[1], Config.IMAGE_SIZE[0], 1), dtype=np.float32)
            dummy = cv2.resize(dummy, Config.IMAGE_SIZE)
            dummy = np.expand_dims(dummy, axis=-1)
            if self.transform:
                dummy = self.transform(dummy)
            return dummy.to(torch.float32), torch.tensor([self.label], dtype=torch.float32), filename
