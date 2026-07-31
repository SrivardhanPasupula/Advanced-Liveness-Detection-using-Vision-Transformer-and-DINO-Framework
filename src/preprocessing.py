"""
Face detection and preprocessing module
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import torch
from torchvision import transforms

class FaceDetector:
    """Face detection using Haar Cascade or DNN"""
    
    def __init__(self, method='haar'):
        self.method = method
        
        if method == 'haar':
            # Load Haar Cascade classifier
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        elif method == 'dnn':
            # Load DNN face detector (more accurate)
            model_file = "res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "deploy.prototxt"
            try:
                self.detector = cv2.dnn.readNetFromCaffe(config_file, model_file)
            except:
                print("DNN model not found, falling back to Haar Cascade")
                self.method = 'haar'
                self.detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Bounding box (x, y, w, h) or None if no face detected
        """
        if self.method == 'haar':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Return largest face
                return max(faces, key=lambda f: f[2] * f[3])
        
        elif self.method == 'dnn':
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            self.detector.setInput(blob)
            detections = self.detector.forward()
            
            # Find detection with highest confidence
            max_conf = 0
            best_box = None
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5 and confidence > max_conf:
                    max_conf = confidence
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    best_box = (x1, y1, x2 - x1, y2 - y1)
            
            return best_box
        
        return None


class FacePreprocessor:
    """Preprocess face images for ViT input"""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.face_detector = FaceDetector(method='haar')
        
        # Normalization for ViT (ImageNet stats)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image: np.ndarray, bbox: Optional[Tuple] = None) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (BGR format)
            bbox: Optional bounding box (x, y, w, h)
            
        Returns:
            Preprocessed tensor
        """
        # Detect face if bbox not provided
        if bbox is None:
            bbox = self.face_detector.detect_face(image)
            if bbox is None:
                # If no face detected, use center crop
                h, w = image.shape[:2]
                size = min(h, w)
                x = (w - size) // 2
                y = (h - size) // 2
                bbox = (x, y, size, size)
        
        # Extract face region
        x, y, w, h = bbox
        face = image[y:y+h, x:x+w]
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        tensor = self.transform(face_rgb)
        
        return tensor
    
    def preprocess_batch(self, images: list) -> torch.Tensor:
        """Preprocess batch of images"""
        tensors = [self.preprocess(img) for img in images]
        return torch.stack(tensors)


def get_augmentation_transform(image_size: int = 224):
    """Get augmentation transforms for training"""
    from albumentations import (
        Compose, HorizontalFlip, RandomBrightnessContrast,
        GaussNoise, MotionBlur, Rotate, ShiftScaleRotate,
        Normalize, Resize
    )
    from albumentations.pytorch import ToTensorV2
    
    return Compose([
        Resize(image_size, image_size),
        HorizontalFlip(p=0.5),
        Rotate(limit=15, p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        MotionBlur(blur_limit=3, p=0.3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
