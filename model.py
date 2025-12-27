"""
Model Module
============
Contains the neural network architecture for multi-label skin classification.
"""

import torch
import torch.nn as nn
import timm  # PyTorch Image Models library for pretrained models

from config import Config


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


def create_model():
    """
    Create and return the model moved to the configured device.
    
    Returns:
        nn.Module: The initialized model on the correct device
    """
    print("\n--- Initializing Model ---")
    
    model = MultiLabelSkinClassifier(
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED
    )
    model = model.to(Config.DEVICE)
    
    return model


def load_model(checkpoint_path=None):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
    
    Returns:
        nn.Module: The loaded model
    """
    checkpoint_path = checkpoint_path or Config.MODEL_SAVE_PATH
    
    model = MultiLabelSkinClassifier(
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=False  # Don't need pretrained weights, loading from checkpoint
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path} (epoch {checkpoint['epoch'] + 1})")
    
    return model
