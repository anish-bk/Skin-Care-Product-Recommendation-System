"""
Prediction Module
=================
Contains utilities for making predictions on new images.
"""

import torch
from PIL import Image
import numpy as np

from config import Config
from dataset import get_transforms
from model import load_model


def predict_single_image(image_path, model=None, threshold=0.5):
    """
    Predict skin problems for a single image.
    
    Args:
        image_path (str): Path to the image file
        model (nn.Module, optional): Trained model. If None, loads from checkpoint.
        threshold (float): Probability threshold for positive prediction
    
    Returns:
        dict: Dictionary with label names as keys and (probability, prediction) as values
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    model.eval()
    
    # Load and preprocess image
    transform = get_transforms(is_training=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
    
    # Make prediction
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Create results dictionary
    results = {}
    for i, label in enumerate(Config.LABEL_COLUMNS):
        prob = probs[i]
        pred = 1 if prob >= threshold else 0
        results[label] = {
            'probability': float(prob),
            'prediction': pred
        }
    
    return results


def predict_batch(image_paths, model=None, threshold=0.5):
    """
    Predict skin problems for multiple images.
    
    Args:
        image_paths (list): List of paths to image files
        model (nn.Module, optional): Trained model. If None, loads from checkpoint.
        threshold (float): Probability threshold for positive prediction
    
    Returns:
        list: List of dictionaries with predictions for each image
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    model.eval()
    
    # Preprocess all images
    transform = get_transforms(is_training=False)
    images = []
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        images.append(transform(image))
    
    image_batch = torch.stack(images).to(Config.DEVICE)
    
    # Make predictions
    with torch.no_grad():
        logits = model(image_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # Create results list
    all_results = []
    for i, path in enumerate(image_paths):
        results = {'image_path': path}
        for j, label in enumerate(Config.LABEL_COLUMNS):
            prob = probs[i, j]
            pred = 1 if prob >= threshold else 0
            results[label] = {
                'probability': float(prob),
                'prediction': pred
            }
        all_results.append(results)
    
    return all_results


def get_detected_problems(results, threshold=0.5):
    """
    Get list of detected skin problems from prediction results.
    
    Args:
        results (dict): Results from predict_single_image
        threshold (float): Probability threshold (already applied in predict)
    
    Returns:
        list: List of tuples (label_name, probability) for detected problems
    """
    detected = []
    for label, info in results.items():
        if info['prediction'] == 1:
            detected.append((label, info['probability']))
    
    # Sort by probability (highest first)
    detected.sort(key=lambda x: x[1], reverse=True)
    
    return detected


def print_prediction_results(results):
    """Pretty print prediction results."""
    print("\n" + "=" * 50)
    print("SKIN PROBLEM DETECTION RESULTS")
    print("=" * 50)
    
    detected = get_detected_problems(results)
    
    if detected:
        print("\n✓ Detected Problems:")
        for label, prob in detected:
            print(f"  • {label}: {prob:.1%}")
    else:
        print("\n✓ No skin problems detected (normal skin)")
    
    print("\n--- All Probabilities ---")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['probability'], reverse=True)
    for label, info in sorted_results:
        marker = "●" if info['prediction'] == 1 else "○"
        print(f"  {marker} {label:20s}: {info['probability']:.1%}")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Analyzing image: {image_path}")
        
        results = predict_single_image(image_path)
        print_prediction_results(results)
    else:
        print("Usage: python predict.py <image_path>")
        print("\nExample: python predict.py test_image.jpg")
