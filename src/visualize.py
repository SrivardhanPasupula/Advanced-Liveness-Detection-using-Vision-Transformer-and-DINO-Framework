"""
Visualization utilities for Face Anti-Spoofing
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional


def visualize_attention_maps(model, image: torch.Tensor, layer_idx: int = -1, 
                             head_idx: Optional[int] = None, save_path: str = None):
    """
    Visualize attention maps from transformer
    
    Args:
        model: Trained model
        image: Input image tensor (1, 3, H, W)
        layer_idx: Which transformer layer to visualize
        head_idx: Which attention head (None = average all heads)
        save_path: Path to save visualization
    """
    model.eval()
    
    # Hook to capture attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        # Capture attention weights from multi-head attention
        attention_weights.append(output)
    
    # Register hook on specific layer
    target_layer = model.student_backbone.blocks[layer_idx].attn
    hook = target_layer.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(image, return_features=False)
    
    hook.remove()
    
    # Process attention weights
    # Shape: (batch, num_heads, num_patches+1, num_patches+1)
    attn = attention_weights[0]
    
    if head_idx is not None:
        attn = attn[0, head_idx]  # Specific head
    else:
        attn = attn[0].mean(0)  # Average over heads
    
    # Get attention from CLS token to patches
    cls_attn = attn[0, 1:]  # Skip CLS token itself
    
    # Reshape to 2D grid
    num_patches = int(np.sqrt(len(cls_attn)))
    attn_map = cls_attn.reshape(num_patches, num_patches).cpu().numpy()
    
    # Visualize
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    img_np = image[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + 
              np.array([0.485, 0.456, 0.406]))
    img_np = np.clip(img_np, 0, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # Attention map
    plt.subplot(1, 2, 2)
    plt.imshow(attn_map, cmap='hot', interpolation='bilinear')
    plt.title(f'Attention Map (Layer {layer_idx})')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention map saved to {save_path}")
    else:
        plt.show()


def visualize_predictions_grid(model, dataloader, device, num_samples: int = 16,
                               save_path: str = 'predictions_grid.png'):
    """
    Visualize grid of predictions
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        num_samples: Number of samples to show
        save_path: Path to save visualization
    """
    model.eval()
    
    images_list = []
    labels_list = []
    preds_list = []
    confs_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images, return_features=False)
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            images_list.extend(images.cpu())
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(predicted.cpu().numpy())
            confs_list.extend(probs[:, 1].cpu().numpy())
            
            if len(images_list) >= num_samples:
                break
    
    # Create grid
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    
    for idx in range(min(num_samples, len(images_list))):
        ax = axes[idx]
        
        # Denormalize image
        img = images_list[idx].permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + 
               np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        # Labels
        true_label = 'Live' if labels_list[idx] == 1 else 'Spoof'
        pred_label = 'Live' if preds_list[idx] == 1 else 'Spoof'
        conf = confs_list[idx]
        
        # Color based on correctness
        color = 'green' if labels_list[idx] == preds_list[idx] else 'red'
        
        ax.set_title(f'True: {true_label}\nPred: {pred_label} ({conf:.2f})',
                    color=color, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Predictions grid saved to {save_path}")


def visualize_feature_space(model, dataloader, device, method='tsne',
                            save_path: str = 'feature_space.png'):
    """
    Visualize feature space using t-SNE or PCA
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        method: 'tsne' or 'pca'
        save_path: Path to save visualization
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    model.eval()
    
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model.student_backbone(images)
            
            features_list.append(features.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    features = np.vstack(features_list)
    labels = np.array(labels_list)
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    features_2d = reducer.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    for label, name, color in [(0, 'Spoof', 'red'), (1, 'Live', 'green')]:
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=color, label=name, alpha=0.6, s=50)
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Feature Space Visualization ({method.upper()})')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature space visualization saved to {save_path}")


def plot_training_history(history: dict, save_path: str = 'training_history.png'):
    """
    Plot training history
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")


# Example usage
if __name__ == '__main__':
    import sys
    sys.path.append('..')
    
    from model_vit_dino import create_model
    from dataset import get_dataloader
    from config import Config
    
    # Load model
    model = create_model(Config).to(Config.DEVICE)
    checkpoint = torch.load('../models/best_model.pth', map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data
    test_loader = get_dataloader('casia', Config.DATA_ROOT, 'test', batch_size=32)
    
    # Visualize predictions
    visualize_predictions_grid(model, test_loader, Config.DEVICE)
    
    # Visualize feature space
    visualize_feature_space(model, test_loader, Config.DEVICE, method='tsne')
