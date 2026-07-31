"""
Vision Transformer with DINO Framework for Face Anti-Spoofing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    
    def __init__(self, image_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Convolutional projection
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """MLP block with GELU activation"""
    
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer backbone"""
    
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Return class token
        return x[:, 0]


class DINOHead(nn.Module):
    """DINO projection head"""
    
    def __init__(self, in_dim: int, out_dim: int = 65536, hidden_dim: int = 2048,
                 bottleneck_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class ViTDINOAntispoofing(nn.Module):
    """
    Complete ViT + DINO model for Face Anti-Spoofing
    Combines supervised classification with self-supervised learning
    """
    
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, num_classes: int = 2):
        super().__init__()
        
        # Student network (trainable)
        self.student_backbone = VisionTransformer(
            image_size, patch_size, 3, embed_dim, depth, num_heads, mlp_ratio
        )
        
        # Teacher network (EMA of student)
        self.teacher_backbone = VisionTransformer(
            image_size, patch_size, 3, embed_dim, depth, num_heads, mlp_ratio
        )
        
        # Freeze teacher
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        
        # DINO projection heads
        self.student_head = DINOHead(embed_dim, out_dim=65536)
        self.teacher_head = DINOHead(embed_dim, out_dim=65536)
        
        # Freeze teacher head
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass
        
        Args:
            x: Input images
            return_features: If True, return features for DINO loss
            
        Returns:
            Classification logits (and optionally DINO features)
        """
        # Student forward
        student_features = self.student_backbone(x)
        logits = self.classifier(student_features)
        
        if return_features:
            # Teacher forward (no gradient)
            with torch.no_grad():
                teacher_features = self.teacher_backbone(x)
            
            # DINO projections
            student_proj = self.student_head(student_features)
            teacher_proj = self.teacher_head(teacher_features)
            
            return logits, student_proj, teacher_proj
        
        return logits
    
    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996):
        """Update teacher network using EMA"""
        for param_student, param_teacher in zip(
            self.student_backbone.parameters(), 
            self.teacher_backbone.parameters()
        ):
            param_teacher.data = momentum * param_teacher.data + (1 - momentum) * param_student.data
        
        for param_student, param_teacher in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters()
        ):
            param_teacher.data = momentum * param_teacher.data + (1 - momentum) * param_student.data


def create_model(config) -> ViTDINOAntispoofing:
    """Create model from config"""
    model = ViTDINOAntispoofing(
        image_size=config.IMAGE_SIZE,
        patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM,
        depth=config.DEPTH,
        num_heads=config.NUM_HEADS,
        mlp_ratio=config.MLP_RATIO,
        num_classes=config.NUM_CLASSES
    )
    return model


# Test model
if __name__ == '__main__':
    model = ViTDINOAntispoofing()
    x = torch.randn(2, 3, 224, 224)
    
    # Test forward pass
    logits = model(x)
    print(f"Logits shape: {logits.shape}")
    
    # Test with DINO features
    logits, student_proj, teacher_proj = model(x, return_features=True)
    print(f"Student projection shape: {student_proj.shape}")
    print(f"Teacher projection shape: {teacher_proj.shape}")
