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
    python main.py --data-dir /path/to/dataset --epochs 20 --batch-size 32
    
    # Full example with all options:
    python main.py --data-dir ./Skin-Problem-Detection-Multiple-Dataset \\
                   --model efficientnet_b0 \\
                   --batch-size 32 \\
                   --epochs 20 \\
                   --lr 0.0001 \\
                   --image-size 224 \\
                   --save-path best_model.pth
"""

import warnings
import torch.nn as nn

from config import Config, parse_args
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
    1. Parse command-line arguments and update config
    2. Setup: Initialize device, create data loaders
    3. Model: Load pretrained model and modify for multi-label
    4. Training: Train for multiple epochs with validation
    5. Evaluation: Final evaluation on test set with detailed metrics
    """
    
    # Parse command-line arguments and update config
    args = parse_args()
    Config.update_from_args(args)
    
    print("=" * 60)
    print("MULTI-LABEL SKIN PROBLEM CLASSIFICATION")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Dataset path:   {Config.DATA_DIR}")
    print(f"  Device:         {Config.DEVICE}")
    print(f"  Model:          {Config.MODEL_NAME}")
    print(f"  Pretrained:     {Config.PRETRAINED}")
    print(f"  Number labels:  {Config.NUM_CLASSES}")
    print(f"  Batch size:     {Config.BATCH_SIZE}")
    print(f"  Epochs:         {Config.NUM_EPOCHS}")
    print(f"  Learning rate:  {Config.LEARNING_RATE}")
    print(f"  Image size:     {Config.IMAGE_SIZE}")
    print(f"  Save path:      {Config.MODEL_SAVE_PATH}")
    
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
