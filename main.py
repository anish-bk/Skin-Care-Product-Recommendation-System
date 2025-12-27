"""
Main Entry Point
================
Orchestrates the complete training and evaluation pipeline.

Multi-Label Skin Problem Classification using PyTorch
------------------------------------------------------
This project fine-tunes a pretrained CNN (EfficientNet-B0) for multi-label
classification of skin problems from images.

Dependencies:
    pip install torch torchvision pandas scikit-learn pillow tqdm timm

Usage:
    python main.py
"""

import warnings
import torch.nn as nn

from config import Config
from dataset import create_data_loaders
from model import create_model
from train import train_model
from evaluate import evaluate_model

warnings.filterwarnings('ignore')


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
    
    print("=" * 60)
    print("MULTI-LABEL SKIN PROBLEM CLASSIFICATION")
    print("=" * 60)
    
    print(f"\nUsing device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Number of labels: {Config.NUM_CLASSES}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    
    # Step 1: Create Data Loaders
    train_loader, valid_loader, test_loader = create_data_loaders()
    
    # Step 2: Create Model
    model = create_model()
    
    # Step 3: Train Model
    model = train_model(model, train_loader, valid_loader)
    
    # Step 4: Evaluate on Test Set
    criterion = nn.BCEWithLogitsLoss()
    test_metrics = evaluate_model(model, test_loader, criterion, Config.DEVICE)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best model saved to: {Config.MODEL_SAVE_PATH}")
    
    return model, test_metrics


if __name__ == "__main__":
    main()
