"""
Configuration Module
====================
Contains all configuration parameters for the skin problem classification system.
"""

import os
import argparse
import torch


class Config:
    """Configuration class for training parameters."""
    
    # Dataset paths - can be overridden via command line
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
    
    @classmethod
    def update_from_args(cls, args):
        """Update configuration from parsed command-line arguments."""
        if args.data_dir:
            cls.DATA_DIR = args.data_dir
            cls.TRAIN_DIR = os.path.join(cls.DATA_DIR, "train")
            cls.VALID_DIR = os.path.join(cls.DATA_DIR, "valid")
            cls.TEST_DIR = os.path.join(cls.DATA_DIR, "test")
        
        if args.model:
            cls.MODEL_NAME = args.model
        
        if args.batch_size:
            cls.BATCH_SIZE = args.batch_size
        
        if args.epochs:
            cls.NUM_EPOCHS = args.epochs
        
        if args.lr:
            cls.LEARNING_RATE = args.lr
        
        if args.image_size:
            cls.IMAGE_SIZE = args.image_size
        
        if args.save_path:
            cls.MODEL_SAVE_PATH = args.save_path
        
        if args.no_pretrained:
            cls.PRETRAINED = False
        
        if args.cpu:
            cls.DEVICE = torch.device("cpu")


def parse_args():
    """
    Parse command-line arguments for training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Multi-Label Skin Problem Classification Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default=None,
        help="Path to dataset directory containing train/valid/test folders"
    )
    
    # Model arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        choices=["efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
                 "resnet50", "resnet101", "vit_base_patch16_224", "vit_small_patch16_224"],
        help="Pretrained model architecture to use"
    )
    
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not use pretrained weights"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input image size (height and width)"
    )
    
    # Output arguments
    parser.add_argument(
        "--save-path", "-o",
        type=str,
        default=None,
        help="Path to save the best model checkpoint"
    )
    
    # Device arguments
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )
    
    return parser.parse_args()
