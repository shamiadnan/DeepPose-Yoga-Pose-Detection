"""
DeepPose 2.0 — Model Evaluation
================================
Generates comprehensive evaluation metrics and visualizations
for the trained yoga pose classification model.
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                   class_names: List[str], 
                   device: str = None) -> Dict:
    """Run full evaluation on test set and compute all metrics.
    
    Args:
        model: Trained PyTorch model in eval mode
        test_loader: Test DataLoader
        class_names: List of class name strings
        device: Computing device (default: from config)
    
    Returns:
        Dictionary containing:
        - 'accuracy': Overall accuracy
        - 'precision': Per-class precision array
        - 'recall': Per-class recall array  
        - 'f1': Per-class F1-score array
        - 'confusion_matrix': Confusion matrix array
        - 'report': Full classification report string
    """
    if device is None:
        device = config.DEVICE
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )
    
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"\n{report}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'report': report
    }
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          save_path: str = None) -> None:
    """Generate and save confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array from sklearn
        class_names: List of class name strings
        save_path: Path to save plot (default: outputs/confusion_matrix.png)
    """
    if save_path is None:
        save_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Confusion Matrix — DeepPose 2.0', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] Confusion matrix saved to: {save_path}")


def plot_training_curves(history: Dict[str, List[float]], 
                         save_dir: str = None) -> None:
    """Generate and save training/validation accuracy and loss curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_dir: Directory to save plots (default: config.OUTPUT_DIR)
    """
    if save_dir is None:
        save_dir = config.OUTPUT_DIR
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # ---- Loss Curves ----
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', markersize=3)
    plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', markersize=3)
    plt.title('Training & Validation Loss — DeepPose 2.0', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] Loss curve saved to: {loss_path}")
    
    # ---- Accuracy Curves ----
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', markersize=3)
    plt.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy', markersize=3)
    plt.title('Training & Validation Accuracy — DeepPose 2.0', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    acc_path = os.path.join(save_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] Accuracy curve saved to: {acc_path}")
