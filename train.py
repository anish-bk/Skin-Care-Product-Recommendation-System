"""
Training Module
===============
Contains training and validation functions for the model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
from evaluate import compute_metrics


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


def train_model(model, train_loader, valid_loader):
    """
    Complete training loop with validation and model checkpointing.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        valid_loader: Validation data loader
    
    Returns:
        model: The trained model (best checkpoint loaded)
    """
    # BCEWithLogitsLoss is the standard loss for multi-label classification
    # It combines Sigmoid + BCELoss for numerical stability
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
            }, Config.MODEL_SAVE_PATH)
            print(f"✓ Saved best model (Macro F1: {best_val_f1:.4f})")
    
    # Load best model
    checkpoint = torch.load(Config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")
    
    return model
