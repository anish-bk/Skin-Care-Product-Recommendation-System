"""
Multi-Label Skin Problem Classification using PyTorch
======================================================

This script fine-tunes a pretrained CNN (EfficientNet-B0) for multi-label
classification of skin problems from images.

Dependencies:
    pip install torch torchvision pandas scikit-learn pillow tqdm timm

Author: Skin Care Product Recommendation System
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm  # PyTorch Image Models library for pretrained models
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration class for training parameters."""
    
    # Dataset paths - UPDATE THIS TO YOUR DATASET LOCATION
    DATA_DIR = "Skin-Problem-Detection-Multiple-Dataset"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VALID_DIR = os.path.join(DATA_DIR, "valid")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    # Model configuration
    MODEL_NAME = "efficientnet_b0"  # Options: resnet50, efficientnet_b0, vit_base_patch16_224
    NUM_CLASSES = 21  # Number of skin problem labels
    PRETRAINED = True
    
    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Image preprocessing
    IMAGE_SIZE = 224
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Label columns (order matters - must match CSV columns)
    LABEL_COLUMNS = [
        'Enlarged Pores', 'acne', 'acne marks', 'acne scar', 'blackhead',
        'burned-skin', 'dark circle', 'darkspot', 'dry', 'freckle',
        'melasma', 'nodules', 'normal skin', 'oily', 'papules',
        'pores', 'pustules', 'skinredness', 'vascular', 'whitehead', 'wrinkle'
    ]

# ============================================================================
# Dataset Class
# ============================================================================

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
        
        # Load CSV file with labels
        csv_path = os.path.join(img_dir, "_classes.csv")
        self.df = pd.read_csv(csv_path)
        
        # Clean filename column (handle potential whitespace)
        self.df['filename'] = self.df['filename'].str.strip()
        
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

# ============================================================================
# Data Augmentation and Preprocessing
# ============================================================================

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

# ============================================================================
# Model Definition
# ============================================================================

class MultiLabelSkinClassifier(nn.Module):
    """
    Multi-label classifier using a pretrained backbone with custom head.
    
    Architecture:
    -------------
    1. Pretrained backbone (EfficientNet/ResNet/ViT) extracts features
    2. Global average pooling reduces spatial dimensions
    3. Dropout for regularization
    4. Linear layer maps to num_classes outputs (one per label)
    
    Key Modification for Multi-Label:
    ---------------------------------
    - Output layer has NO softmax/sigmoid (raw logits)
    - BCEWithLogitsLoss applies sigmoid internally for numerical stability
    - Each output neuron predicts independently (not mutually exclusive)
    
    Args:
        model_name (str): Name of the pretrained model from timm
        num_classes (int): Number of output labels
        pretrained (bool): Whether to use pretrained weights
    """
    
    def __init__(self, model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES, pretrained=True):
        super().__init__()
        
        # Load pretrained model from timm library
        # num_classes=0 removes the original classification head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get the feature dimension from the backbone
        # This varies by model (e.g., ResNet50=2048, EfficientNet-B0=1280, ViT=768)
        feature_dim = self.backbone.num_features
        
        # Custom classification head for multi-label classification
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, num_classes)
            # NO activation here - BCEWithLogitsLoss handles sigmoid
        )
        
        print(f"Created {model_name} with {feature_dim} features -> {num_classes} labels")
    
    def forward(self, x):
        """
        Forward pass: image -> features -> logits
        
        Args:
            x (Tensor): Input images of shape (B, C, H, W)
        
        Returns:
            logits (Tensor): Raw output scores of shape (B, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)  # (B, feature_dim)
        
        # Classify
        logits = self.classifier(features)  # (B, num_classes)
        
        return logits

# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Training Loop Explanation:
    --------------------------
    1. Set model to training mode (enables dropout, batch norm updates)
    2. For each batch:
       a. Move data to device (GPU/CPU)
       b. Zero gradients from previous iteration
       c. Forward pass to get predictions
       d. Compute loss (BCEWithLogitsLoss for multi-label)
       e. Backward pass to compute gradients
       f. Optimizer step to update weights
    3. Track running loss for monitoring
    
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for images, labels in progress_bar:
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss (BCEWithLogitsLoss handles sigmoid internally)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track loss
        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    """
    Validate the model on a dataset.
    
    Validation differs from training:
    ---------------------------------
    1. model.eval() disables dropout and uses running stats for batch norm
    2. torch.no_grad() prevents gradient computation (saves memory)
    3. We collect predictions for metric computation
    
    Returns:
        tuple: (average_loss, all_predictions, all_labels)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Apply sigmoid to get probabilities, then threshold at 0.5
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Collect for metrics
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            running_loss += loss.item() * images.size(0)
    
    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, all_preds, all_labels

# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(y_true, y_pred, label_names):
    """
    Compute multi-label classification metrics.
    
    Metrics Explanation:
    --------------------
    1. Per-Label Accuracy: Accuracy for each label independently
    2. Micro F1: Aggregates TP/FP/FN across all labels, then computes F1
       - Good when you care about overall performance
    3. Macro F1: Computes F1 for each label, then averages
       - Treats all labels equally (important for imbalanced datasets)
    4. Samples F1: Computes F1 for each sample, then averages
       - Good for multi-label problems
    
    Args:
        y_true (ndarray): Ground truth labels (N, num_classes)
        y_pred (ndarray): Predicted labels (N, num_classes)
        label_names (list): Names of each label for reporting
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    metrics = {}
    
    # Per-label accuracy
    per_label_acc = []
    for i, name in enumerate(label_names):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        per_label_acc.append(acc)
        metrics[f'acc_{name}'] = acc
    
    metrics['mean_per_label_accuracy'] = np.mean(per_label_acc)
    
    # F1 scores
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['samples_f1'] = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    # Exact match ratio (all labels correct for a sample)
    exact_match = np.all(y_true == y_pred, axis=1).mean()
    metrics['exact_match_ratio'] = exact_match
    
    return metrics

def print_metrics(metrics, label_names):
    """Pretty print the metrics."""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print("\n--- Overall Metrics ---")
    print(f"Micro F1 Score:        {metrics['micro_f1']:.4f}")
    print(f"Macro F1 Score:        {metrics['macro_f1']:.4f}")
    print(f"Samples F1 Score:      {metrics['samples_f1']:.4f}")
    print(f"Exact Match Ratio:     {metrics['exact_match_ratio']:.4f}")
    print(f"Mean Per-Label Acc:    {metrics['mean_per_label_accuracy']:.4f}")
    
    print("\n--- Per-Label Accuracy ---")
    for name in label_names:
        print(f"  {name:20s}: {metrics[f'acc_{name}']:.4f}")

# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """
    Main function to orchestrate the training pipeline.
    
    Pipeline Steps:
    ---------------
    1. Setup: Initialize device, create data loaders
    2. Model: Load pretrained model and modify for multi-label
    3. Training: Train for multiple epochs with validation
    4. Evaluation: Final evaluation on test set with detailed metrics
    """
    
    print(f"Using device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Number of labels: {Config.NUM_CLASSES}")
    
    # -------------------------------------------------------------------------
    # Step 1: Create Datasets and DataLoaders
    # -------------------------------------------------------------------------
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
    
    # -------------------------------------------------------------------------
    # Step 2: Create Model, Loss Function, and Optimizer
    # -------------------------------------------------------------------------
    print("\n--- Initializing Model ---")
    
    model = MultiLabelSkinClassifier(
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED
    )
    model = model.to(Config.DEVICE)
    
    # BCEWithLogitsLoss is the standard loss for multi-label classification
    # It combines Sigmoid + BCELoss for numerical stability
    # pos_weight can be used for class imbalance (optional)
    criterion = nn.BCEWithLogitsLoss()
    
    # AdamW optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler (optional but helps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.NUM_EPOCHS,
        eta_min=1e-6
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Training Loop
    # -------------------------------------------------------------------------
    print("\n--- Starting Training ---")
    
    best_val_f1 = 0.0
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # Validate
        val_loss, val_preds, val_labels = validate(model, valid_loader, criterion, Config.DEVICE)
        
        # Compute validation metrics
        val_metrics = compute_metrics(val_labels, val_preds, Config.LABEL_COLUMNS)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Micro F1: {val_metrics['micro_f1']:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model based on macro F1 (accounts for class imbalance)
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': best_val_f1,
            }, 'best_skin_classifier.pth')
            print(f"✓ Saved best model (Macro F1: {best_val_f1:.4f})")
    
    # -------------------------------------------------------------------------
    # Step 4: Final Evaluation on Test Set
    # -------------------------------------------------------------------------
    print("\n--- Final Evaluation on Test Set ---")
    
    # Load best model
    checkpoint = torch.load('best_skin_classifier.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
    
    # Evaluate on test set
    test_loss, test_preds, test_labels = validate(model, test_loader, criterion, Config.DEVICE)
    test_metrics = compute_metrics(test_labels, test_preds, Config.LABEL_COLUMNS)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print_metrics(test_metrics, Config.LABEL_COLUMNS)
    
    # Print sklearn classification report
    print("\n--- Detailed Classification Report ---")
    print(classification_report(
        test_labels,
        test_preds,
        target_names=Config.LABEL_COLUMNS,
        zero_division=0
    ))
    
    print("\nTraining Complete!")
    return model, test_metrics


if __name__ == "__main__":
    main()
