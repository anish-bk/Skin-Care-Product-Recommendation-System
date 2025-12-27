"""
Dataset Module
==============
Contains the custom Dataset class and data augmentation utilities.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import Config


class SkinProblemDataset(Dataset):
    """
    Custom PyTorch Dataset for multi-label skin problem classification.
    
    This dataset loads images and their corresponding multi-label annotations
    from a directory structure with CSV files containing binary labels.
    
    Key Design Decisions:
    ---------------------
    1. Labels are loaded from CSV and converted to float tensors for BCEWithLogitsLoss
    2. Images are loaded on-demand to manage memory efficiently
    3. Transforms are applied during __getitem__ for data augmentation
    
    Args:
        img_dir (str): Path to directory containing images and _classes.csv
        transform (callable, optional): Transform to apply to images
        label_columns (list): List of label column names in the CSV
    """
    
    def __init__(self, img_dir, transform=None, label_columns=None):
        self.img_dir = img_dir
        self.transform = transform
        self.label_columns = label_columns or Config.LABEL_COLUMNS
        
        # Load CSV file with labels (auto-detect delimiter for TSV/CSV)
        csv_path = os.path.join(img_dir, "_classes.csv")
        self.df = pd.read_csv(csv_path, sep=None, engine='python')  # Auto-detect separator
        self.df.columns = self.df.columns.str.strip()
        
        # Clean filename column (handle potential whitespace)
        self.df['filename'] = self.df['filename'].str.strip()
        
        # Debug: print actual columns found
        print(f"Found columns: {list(self.df.columns)}")
        
        # Verify all label columns exist
        missing_cols = set(self.label_columns) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing label columns in CSV: {missing_cols}")
        
        print(f"Loaded {len(self.df)} samples from {img_dir}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Preprocessed image tensor of shape (C, H, W)
            labels (Tensor): Multi-label tensor of shape (num_classes,) with values 0.0 or 1.0
        """
        # Get image filename and construct full path
        img_name = self.df.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image and convert to RGB (handles grayscale/RGBA images)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Extract multi-label vector (convert to float for BCE loss)
        labels = self.df.iloc[idx][self.label_columns].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, labels


def get_transforms(is_training=True):
    """
    Returns image transforms for training or validation/testing.
    
    Training Augmentations:
    -----------------------
    - RandomResizedCrop: Scale and crop for spatial invariance
    - RandomHorizontalFlip: Mirror images (skin can appear on either side)
    - ColorJitter: Adjust brightness/contrast (lighting variations)
    - RandomRotation: Slight rotation for orientation invariance
    
    Validation/Test:
    ----------------
    - Simple resize and center crop for consistent evaluation
    
    All transforms normalize using ImageNet statistics (required for pretrained models).
    """
    
    # ImageNet normalization statistics
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE + 32),  # Slightly larger
            transforms.CenterCrop(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ])


def create_data_loaders():
    """
    Create and return train, validation, and test data loaders.
    
    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    print("\n--- Loading Datasets ---")
    
    train_dataset = SkinProblemDataset(
        Config.TRAIN_DIR,
        transform=get_transforms(is_training=True),
        label_columns=Config.LABEL_COLUMNS
    )
    
    valid_dataset = SkinProblemDataset(
        Config.VALID_DIR,
        transform=get_transforms(is_training=False),
        label_columns=Config.LABEL_COLUMNS
    )
    
    test_dataset = SkinProblemDataset(
        Config.TEST_DIR,
        transform=get_transforms(is_training=False),
        label_columns=Config.LABEL_COLUMNS
    )
    
    # DataLoaders with shuffling for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader
