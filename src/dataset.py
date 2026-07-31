"""
Dataset loaders for Face Anti-Spoofing
"""

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
from src.preprocessing import FacePreprocessor, get_augmentation_transform


class FaceAntispoofingDataset(Dataset):
    """Generic Face Anti-Spoofing Dataset"""
    
    def __init__(self, root_dir: str, transform=None, augment: bool = False):
        """
        Args:
            root_dir: Root directory with 'live' and 'spoof' subdirectories
            transform: Optional transform to apply
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        
        self.samples = []
        self.labels = []
        
        # Load live samples (label = 1)
        live_dir = os.path.join(root_dir, 'live')
        if os.path.exists(live_dir):
            for img_name in os.listdir(live_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append(os.path.join(live_dir, img_name))
                    self.labels.append(1)
        
        # Load spoof samples (label = 0)
        spoof_dir = os.path.join(root_dir, 'spoof')
        if os.path.exists(spoof_dir):
            for img_name in os.listdir(spoof_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append(os.path.join(spoof_dir, img_name))
                    self.labels.append(0)
        
        print(f"Loaded {len(self.samples)} samples ({sum(self.labels)} live, {len(self.labels) - sum(self.labels)} spoof)")
        
        if self.augment:
            self.aug_transform = get_augmentation_transform()
        else:
            self.preprocessor = FacePreprocessor()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return dummy data if image can't be loaded
            return torch.zeros(3, 224, 224), label
        
        # Apply augmentation or preprocessing
        if self.augment:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = self.aug_transform(image=image_rgb)
            image_tensor = augmented['image']
        else:
            image_tensor = self.preprocessor.preprocess(image)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, label


class CASIAFASDDataset(FaceAntispoofingDataset):
    """CASIA Face Anti-Spoofing Dataset"""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None, augment: bool = False):
        """
        Args:
            root_dir: Root directory of CASIA-FASD dataset
            split: 'train' or 'test'
            transform: Optional transform
            augment: Whether to apply augmentation
        """
        data_dir = os.path.join(root_dir, split)
        super().__init__(data_dir, transform, augment)


class ReplayAttackDataset(FaceAntispoofingDataset):
    """Replay-Attack Dataset"""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None, augment: bool = False):
        """
        Args:
            root_dir: Root directory of Replay-Attack dataset
            split: 'train' or 'test'
            transform: Optional transform
            augment: Whether to apply augmentation
        """
        data_dir = os.path.join(root_dir, split)
        super().__init__(data_dir, transform, augment)


def get_dataloader(dataset_name: str, root_dir: str, split: str = 'train', 
                   batch_size: int = 32, augment: bool = False, 
                   num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    """
    Get dataloader for specified dataset
    
    Args:
        dataset_name: 'casia', 'replay', 'custom', 'custom1', or 'custom2'
        root_dir: Root directory of dataset (should point to train or test folder)
        split: 'train' or 'test'
        batch_size: Batch size
        augment: Whether to apply augmentation
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    # Use generic dataset loader for all datasets
    dataset = FaceAntispoofingDataset(root_dir, augment=augment)
    
    # Set num_workers to 0 on Windows to avoid multiprocessing issues
    import platform
    if platform.system() == 'Windows':
        num_workers = 0
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    return dataloader


# Example usage and testing
if __name__ == '__main__':
    # Test dataset loading
    dataset = FaceAntispoofingDataset('./data/test', augment=True)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"Image shape: {img.shape}, Label: {label}")
