import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from dataset import ImageDataset
from config import Config
import numpy as np
import torch

class QualityModel(nn.Module):
    def __init__(self):
        super(QualityModel, self).__init__()
        # Using EfficientNet instead of ResNet for better performance
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Modify first conv layer for grayscale input
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Modify classifier for binary output
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

# data_utils.py
def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ])

    good_dataset = ImageDataset(root_dir=Config.GOOD_DIR, transform=transform, label=1)
    bad_dataset = ImageDataset(root_dir=Config.BAD_DIR, transform=transform, label=0)

    # Split datasets
    good_train_size = int((1 - Config.VAL_SPLIT) * len(good_dataset))
    good_val_size = len(good_dataset) - good_train_size
    bad_train_size = int((1 - Config.VAL_SPLIT) * len(bad_dataset))
    bad_val_size = len(bad_dataset) - bad_train_size

    good_train, good_val = random_split(good_dataset, [good_train_size, good_val_size])
    bad_train, bad_val = random_split(bad_dataset, [bad_train_size, bad_val_size])

    train_dataset = ConcatDataset([good_train, bad_train])
    val_dataset = ConcatDataset([good_val, bad_val])

    # Create balanced sampler
    train_labels = np.array([int(sample[1].item()) for sample in train_dataset])
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=1)

    return train_loader, val_loader

# trainer.py
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_loss = float('inf')
        self.early_stop_count = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        # Unpack three values (image, label, filename) and ignore the filename
        for inputs, labels, _ in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels, _ in self.val_loader:  # Unpack and ignore the filename
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)