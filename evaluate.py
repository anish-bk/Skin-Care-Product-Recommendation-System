"""
Evaluation Module
=================
Contains metrics computation and evaluation utilities.
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

from config import Config


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


def print_classification_report(y_true, y_pred, label_names):
    """Print sklearn's detailed classification report."""
    print("\n--- Detailed Classification Report ---")
    print(classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0
    ))


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate a model on the test set and print detailed metrics.
    
    Args:
        model: Trained model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to run evaluation on
    
    Returns:
        dict: Test metrics
    """
    from train import validate  # Import here to avoid circular imports
    
    print("\n--- Final Evaluation on Test Set ---")
    
    test_loss, test_preds, test_labels = validate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_labels, test_preds, Config.LABEL_COLUMNS)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print_metrics(test_metrics, Config.LABEL_COLUMNS)
    print_classification_report(test_labels, test_preds, Config.LABEL_COLUMNS)
    
    return test_metrics
