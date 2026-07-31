"""
GPU-Optimized Training for ViT-DINO Face Anti-Spoofing
Features:
- Mixed precision training (FP16)
- Multi-GPU support (DataParallel)
- Gradient accumulation
- Better progress tracking
- TensorBoard logging
- Automatic checkpoint saving
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import argparse
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
        # Supervised classification loss
        supervised_loss = self.ce_loss(logits, labels)
        
        # Self-supervised DINO loss
        dino_loss = self.dino_loss(student_proj, teacher_proj.detach())
        
        # Combined loss
        total_loss = (self.supervised_weight * supervised_loss + 
                     self.dino_weight * dino_loss)
        
        return total_loss, supervised_loss, dino_loss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler, 
                accumulation_steps=1, use_amp=True):
    """Train for one epoch with mixed precision"""
    model.train()
    
    total_loss = 0
    total_supervised = 0
    total_dino = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            logits, student_proj, teacher_proj = model(images, return_features=True)
            loss, sup_loss, dino_loss = criterion(logits, labels, student_proj, teacher_proj)
            loss = loss / accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update teacher network (EMA)
            if hasattr(model, 'module'):
                model.module.update_teacher(momentum=Config.MOMENTUM_TEACHER)
            else:
                model.update_teacher(momentum=Config.MOMENTUM_TEACHER)
        
        # Statistics
        total_loss += loss.item() * accumulation_steps
        total_supervised += sup_loss.item()
        total_dino += dino_loss.item()
        
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_supervised = total_supervised / len(dataloader)
    avg_dino = total_dino / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_supervised, avg_dino, accuracy


@torch.no_grad()
def validate(model, dataloader, criterion, device, use_amp=True):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Validating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            logits = model(images, return_features=False)
            loss = criterion(logits, labels)
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    # Calculate per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    live_acc = 100. * np.sum((all_preds == 0) & (all_labels == 0)) / np.sum(all_labels == 0) if np.sum(all_labels == 0) > 0 else 0
    spoof_acc = 100. * np.sum((all_preds == 1) & (all_labels == 1)) / np.sum(all_labels == 1) if np.sum(all_labels == 1) > 0 else 0
    
    return avg_loss, accuracy, live_acc, spoof_acc


def train(args):
    """Main training function"""
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("⚠ GPU not available, using CPU")
    
    # Create model
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    model = create_model(Config)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"✓ Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1e6:.2f} MB")
    
    # Create dataloaders
    print("\n" + "="*60)
    print(f"Loading {args.dataset} dataset...")
    print("="*60)
    
    # Get dataset path
    if args.dataset in Config.DATASETS:
        dataset_config = Config.DATASETS[args.dataset]
        train_path = os.path.join(Config.DATA_ROOT, dataset_config['train'])
        test_path = os.path.join(Config.DATA_ROOT, dataset_config['test'])
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Check if paths exist
    if not os.path.exists(train_path):
        print(f"❌ Training path not found: {train_path}")
        print(f"\nAvailable datasets in {Config.DATA_ROOT}:")
        if os.path.exists(Config.DATA_ROOT):
            for item in os.listdir(Config.DATA_ROOT):
                print(f"  - {item}")
        sys.exit(1)
    
    train_loader = get_dataloader(
        args.dataset, 
        train_path,
        split='train',
        batch_size=args.batch_size,
        augment=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = get_dataloader(
        args.dataset,
        test_path,
        split='test',
        batch_size=args.batch_size,
        augment=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"✓ Training samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Batch size: {args.batch_size}")
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    
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
        lr=args.lr,
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)
    
    # Training setup
    best_acc = 0
    start_epoch = 1
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Resume from checkpoint
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\n✓ Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('accuracy', 0)
            print(f"  Resuming from epoch {start_epoch}, best acc: {best_acc:.2f}%")
    
    # Training loop
    print("\n" + "="*60)
    print(f"Starting training for {args.epochs} epochs")
    print("="*60)
    print(f"Device: {device}")
    print(f"Mixed Precision: {args.use_amp}")
    print(f"Gradient Accumulation: {args.accumulation_steps} steps")
    print(f"Effective Batch Size: {args.batch_size * args.accumulation_steps}")
    print("="*60 + "\n")
    
    training_start = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, sup_loss, dino_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, 
            scaler, args.accumulation_steps, args.use_amp
        )
        
        print(f"\n📊 Training Results:")
        print(f"  Loss: {train_loss:.4f} | Supervised: {sup_loss:.4f} | DINO: {dino_loss:.4f}")
        print(f"  Accuracy: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc, live_acc, spoof_acc = validate(
            model, val_loader, val_criterion, device, args.use_amp
        )
        
        print(f"\n📊 Validation Results:")
        print(f"  Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")
        print(f"  Live Accuracy: {live_acc:.2f}% | Spoof Accuracy: {spoof_acc:.2f}%")
        
        # Update learning rate
        if args.scheduler == 'cosine':
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        print(f"\n⏱ Epoch time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'live_acc': live_acc,
                'spoof_acc': spoof_acc
            }
            save_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f"✓ Saved best model (acc: {best_acc:.2f}%) to {save_path}")
        
        # Save checkpoint every N epochs
        if epoch % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'live_acc': live_acc,
                'spoof_acc': spoof_acc
            }
            save_path = os.path.join(Config.MODEL_SAVE_PATH, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)
            print(f"✓ Saved checkpoint to {save_path}")
    
    training_time = time.time() - training_start
    
    print(f"\n{'='*60}")
    print("🎉 Training completed!")
    print(f"{'='*60}")
    print(f"Total training time: {training_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {Config.MODEL_SAVE_PATH}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU-Optimized Training for ViT-DINO Face Anti-Spoofing')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='custom1', 
                       choices=['casia', 'replay', 'custom', 'custom1', 'custom2'], 
                       help='Dataset to use')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, 
                       help='Learning rate (default: 1e-4)')
    
    # Optimization
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use mixed precision training (default: True)')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'onecycle'],
                       help='Learning rate scheduler (default: cosine)')
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("GPU-Optimized Face Anti-Spoofing Training")
    print("="*60)
    print(f"Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("="*60 + "\n")
    
    train(args)
