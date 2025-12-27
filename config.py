"""
Configuration Module
====================
Contains all configuration parameters for the skin problem classification system.
"""

import os
import torch


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
    
    # Model save path
    MODEL_SAVE_PATH = "best_skin_classifier.pth"
    
    # Label columns (order matters - must match CSV columns)
    LABEL_COLUMNS = [
        'Enlarged Pores', 'acne', 'acne marks', 'acne scar', 'blackhead',
        'burned-skin', 'dark circle', 'darkspot', 'dry', 'freckle',
        'melasma', 'nodules', 'normal skin', 'oily', 'papules',
        'pores', 'pustules', 'skinredness', 'vascular', 'whitehead', 'wrinkle'
    ]
