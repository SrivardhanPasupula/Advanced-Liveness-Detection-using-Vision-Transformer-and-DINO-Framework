"""
Training pipeline for ViT-DINO Face Anti-Spoofing
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_vit_dino import create_model, ViTDINOAntispoofing
from src.dataset import get_dataloader
from config import Config


class DINOLoss(nn.Module):
    """DINO self-supervised loss"""
    
    def __init__(self, student_temp: float = 0.1, teacher_temp: float = 0.04):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
    
    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """
        Compute DINO loss (cross-entropy between student and teacher)
        
        Args:
            student_output: Student network output
            teacher_output: Teacher network output (detached)
            
        Returns:
            DINO loss value
        """
        # Normalize and apply temperature
        student_out = F.log_softmax(student_output / self.student_temp, dim=-1)
        teacher_out = F.softmax(teacher_output / self.teacher_temp, dim=-1)
        
        # Cross-entropy loss
        loss = -torch.sum(teacher_out * student_out, dim=-1).mean()
        
        return loss


class CombinedLoss(nn.Module):
    """Combined supervised + self-supervised loss"""
    
    def __init__(self, supervised_weight: float = 1.0, dino_weight: float = 0.5,
                 student_temp: float = 0.1, teacher_temp: float = 0.04):
        super().__init__()
        self.supervised_weight = supervised_weight
        self.dino_weight = dino_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.dino_loss = DINOLoss(student_temp, teacher_temp)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                student_proj: torch.Tensor, teacher_proj: torch.Tensor) -> tuple:
        """
        Compute combined loss
        
        Returns:
            total_loss, supervised_loss, dino_loss
        """
        # Supervised classification loss
        supervised_loss = self.ce_loss(logits, labels)
        
        # Self-supervised DINO loss
        dino_loss = self.dino_loss(student_proj, teacher_proj.detach())
        
        # Combined loss
        total_loss = (self.supervised_weight * supervised_loss + 
                     self.dino_weight * dino_loss)
        
        return total_loss, supervised_loss, dino_loss


def train_epoch(model: ViTDINOAntispoofing, dataloader: DataLoader, 
                criterion: CombinedLoss, optimizer, device, epoch: int):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_supervised = 0
    total_dino = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass with DINO features
        logits, student_proj, teacher_proj = model(images, return_features=True)
        
        # Compute loss
        loss, sup_loss, dino_loss = criterion(logits, labels, student_proj, teacher_proj)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update teacher network (EMA)
        model.update_teacher(momentum=Config.MOMENTUM_TEACHER)
        
        # Statistics
        total_loss += loss.item()
        total_supervised += sup_loss.item()
        total_dino += dino_loss.item()
        
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_supervised = total_supervised / len(dataloader)
    avg_dino = total_dino / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_supervised, avg_dino, accuracy


def validate(model: ViTDINOAntispoofing, dataloader: DataLoader, 
             criterion: nn.Module, device):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass (no DINO features needed for validation)
            logits = model(images, return_features=False)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train(args):
    """Main training function"""
    
    # Setup
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = create_model(Config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders
    print(f"Loading {args.dataset} dataset...")
    
    # Get dataset path
    if args.dataset in Config.DATASETS:
        dataset_config = Config.DATASETS[args.dataset]
        train_path = os.path.join(Config.DATA_ROOT, dataset_config['train'])
        test_path = os.path.join(Config.DATA_ROOT, dataset_config['test'])
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    train_loader = get_dataloader(
        args.dataset, 
        train_path,
        split='train',
        batch_size=Config.BATCH_SIZE,
        augment=True
    )
    
    val_loader = get_dataloader(
        args.dataset,
        test_path,
        split='test',
        batch_size=Config.BATCH_SIZE,
        augment=False
    )
    
    # Loss and optimizer
    criterion = CombinedLoss(
        supervised_weight=Config.SUPERVISED_LOSS_WEIGHT,
        dino_weight=Config.DINO_LOSS_WEIGHT,
        student_temp=Config.STUDENT_TEMP,
        teacher_temp=Config.TEACHER_TEMP
    )
    
    val_criterion = nn.CrossEntropyLoss()
    
    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, sup_loss, dino_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        print(f"\nTraining - Loss: {train_loss:.4f} | "
              f"Supervised: {sup_loss:.4f} | DINO: {dino_loss:.4f} | "
              f"Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, val_criterion, device)
        print(f"Validation - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc
            }
            torch.save(checkpoint, os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth'))
            print(f"✓ Saved best model (acc: {best_acc:.2f}%)")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc
            }
            torch.save(checkpoint, os.path.join(Config.MODEL_SAVE_PATH, f'checkpoint_epoch_{epoch}.pth'))
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ViT-DINO Face Anti-Spoofing')
    parser.add_argument('--dataset', type=str, default='custom', 
                       choices=['casia', 'replay', 'custom', 'custom1', 'custom2'], 
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.lr:
        Config.LEARNING_RATE = args.lr
    
    train(args)
