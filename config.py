"""
Configuration file for Face Anti-Spoofing system
"""

import torch

class Config:
    # Model parameters
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    NUM_CLASSES = 2  # Live vs Spoof
    EMBED_DIM = 768
    DEPTH = 12
    NUM_HEADS = 12
    MLP_RATIO = 4.0
    
    # DINO parameters
    TEACHER_TEMP = 0.04
    STUDENT_TEMP = 0.1
    MOMENTUM_TEACHER = 0.996
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 10
    
    # Loss weights
    SUPERVISED_LOSS_WEIGHT = 1.0
    DINO_LOSS_WEIGHT = 0.5
    
    # Data augmentation
    AUGMENTATION = True
    
    # Paths
    DATA_ROOT = './data'
    MODEL_SAVE_PATH = './models'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluation thresholds
    CONFIDENCE_THRESHOLD = 0.5
    
    # Dataset specific
    DATASETS = {
        'casia': {
            'train': 'CASIA-FASD/train',
            'test': 'CASIA-FASD/test'
        },
        'replay': {
            'train': 'Replay-Attack/train',
            'test': 'Replay-Attack/test'
        },
        'custom': {
            'train': 'combined_dataset/train',
            'test': 'combined_dataset/test'
        },
        'custom1': {
            'train': 'custom_dataset1/train',
            'test': 'custom_dataset1/test'
        },
        'custom2': {
            'train': 'custom_dataset2/train',
            'test': 'custom_dataset2/test'
        }
    }
