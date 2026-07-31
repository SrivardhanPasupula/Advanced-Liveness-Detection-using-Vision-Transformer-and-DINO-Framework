"""
Evaluation metrics for Face Anti-Spoofing
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_vit_dino import create_model
from src.dataset import get_dataloader
from config import Config


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: Ground truth labels (0=spoof, 1=live)
        y_pred: Predicted labels
        y_scores: Prediction scores/probabilities
        
    Returns:
        Dictionary of metrics
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # APCER: Attack Presentation Classification Error Rate
    # Proportion of attack presentations incorrectly classified as bona fide
    # APCER = FP / (TN + FP) = FP / Total_Attacks
    apcer = fp / (tn + fp) if (tn + fp) > 0 else 0
    
    # BPCER: Bona Fide Presentation Classification Error Rate
    # Proportion of bona fide presentations incorrectly classified as attacks
    # BPCER = FN / (FN + TP) = FN / Total_Live
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # ACER: Average Classification Error Rate
    acer = (apcer + bpcer) / 2
    
    # Additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # ROC AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        'accuracy': accuracy * 100,
        'apcer': apcer * 100,
        'bpcer': bpcer * 100,
        'acer': acer * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1_score * 100,
        'roc_auc': roc_auc,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    }
    
    return metrics, (fpr, tpr, thresholds)


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset"""
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_scores = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            # Forward pass
            logits = model(images, return_features=False)
            probs = torch.softmax(logits, dim=1)
            
            # Get predictions and scores
            _, predicted = logits.max(1)
            scores = probs[:, 1]  # Probability of being live
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_scores = np.array(all_scores)
    
    return all_labels, all_predictions, all_scores


def plot_roc_curve(fpr, tpr, roc_auc, save_path='roc_curve.png'):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (APCER)')
    plt.ylabel('True Positive Rate (1 - BPCER)')
    plt.title('ROC Curve - Face Anti-Spoofing')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")


def plot_confusion_matrix(cm_dict, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = np.array([[cm_dict['tn'], cm_dict['fp']], 
                   [cm_dict['fn'], cm_dict['tp']]])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Spoof', 'Live']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")


def main(args):
    """Main evaluation function"""
    
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = create_model(Config).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load test data
    print(f"Loading {args.dataset} test dataset...")
    test_loader = get_dataloader(
        args.dataset,
        Config.DATA_ROOT,
        split='test',
        batch_size=Config.BATCH_SIZE,
        augment=False
    )
    
    # Evaluate
    print("\nEvaluating model...")
    y_true, y_pred, y_scores = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    metrics, (fpr, tpr, thresholds) = calculate_metrics(y_true, y_pred, y_scores)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:     {metrics['accuracy']:.2f}%")
    print(f"APCER:        {metrics['apcer']:.2f}%")
    print(f"BPCER:        {metrics['bpcer']:.2f}%")
    print(f"ACER:         {metrics['acer']:.2f}%")
    print(f"Precision:    {metrics['precision']:.2f}%")
    print(f"Recall:       {metrics['recall']:.2f}%")
    print(f"F1-Score:     {metrics['f1_score']:.2f}%")
    print(f"ROC AUC:      {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TN: {cm['tn']:4d}  |  FP: {cm['fp']:4d}")
    print(f"  FN: {cm['fn']:4d}  |  TP: {cm['tp']:4d}")
    print("="*60)
    
    # Plot visualizations
    if args.plot:
        print("\nGenerating plots...")
        plot_roc_curve(fpr, tpr, metrics['roc_auc'], 'roc_curve.png')
        plot_confusion_matrix(metrics['confusion_matrix'], 'confusion_matrix.png')
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ViT-DINO Face Anti-Spoofing')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='custom',
                       choices=['casia', 'replay', 'custom', 'custom1', 'custom2'], 
                       help='Dataset to evaluate on')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    main(args)
