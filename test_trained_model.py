"""
Test your trained model on webcam or images
"""

import cv2
import torch
import numpy as np
import argparse
from src.model_vit_dino import create_model
from src.preprocessing import FacePreprocessor
from config import Config

class TrainedModelTester:
    def __init__(self, model_path):
        """Initialize with trained model"""
        print("Loading trained model...")
        
        # Create model
        self.model = create_model(Config)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=Config.DEVICE, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(Config.DEVICE)
        self.model.eval()
        
        # Preprocessor
        self.preprocessor = FacePreprocessor()
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Model accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%")
        print(f"✓ Device: {Config.DEVICE}")
    
    def predict(self, image):
        """Predict if image is live or spoof"""
        # Preprocess
        img_tensor = self.preprocessor.preprocess(image)
        img_tensor = img_tensor.unsqueeze(0).to(Config.DEVICE)
        
        # Predict
        with torch.no_grad():
            logits = self.model(img_tensor, return_features=False)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        # 0 = spoof, 1 = live
        label = "LIVE" if pred == 1 else "SPOOF"
        
        return label, confidence
    
    def test_webcam(self):
        """Test on webcam"""
        print("\n" + "="*60)
        print("🎥 Webcam Test - Press 'q' to quit")
        print("="*60)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Extract face
                face = frame[y:y+h, x:x+w]
                
                # Predict
                label, confidence = self.predict(face)
                
                # Draw results
                color = (0, 255, 0) if label == "LIVE" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Text
                text = f"{label}: {confidence*100:.1f}%"
                cv2.putText(frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Alert for spoof
                if label == "SPOOF":
                    cv2.putText(frame, "FAKE ALERT!", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Display
            cv2.imshow('Face Anti-Spoofing - Trained Model', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Webcam test completed")
    
    def test_image(self, image_path):
        """Test on single image"""
        print(f"\n📷 Testing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Cannot load image: {image_path}")
            return
        
        # Detect face
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("❌ No face detected")
            return
        
        # Test each face
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y:y+h, x:x+w]
            label, confidence = self.predict(face)
            
            print(f"\n  Face {i+1}:")
            print(f"    Prediction: {label}")
            print(f"    Confidence: {confidence*100:.1f}%")
            
            # Draw on image
            color = (0, 255, 0) if label == "LIVE" else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            text = f"{label}: {confidence*100:.1f}%"
            cv2.putText(image, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Display
        cv2.imshow('Result', image)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Test Trained Model')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--mode', type=str, default='webcam',
                       choices=['webcam', 'image'],
                       help='Test mode')
    parser.add_argument('--input', type=str, default=None,
                       help='Input image path (for image mode)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎯 Trained Model Tester")
    print("="*60)
    
    # Create tester
    tester = TrainedModelTester(args.model)
    
    # Test
    if args.mode == 'webcam':
        tester.test_webcam()
    elif args.mode == 'image':
        if args.input is None:
            print("❌ Please provide --input for image mode")
            return
        tester.test_image(args.input)

if __name__ == '__main__':
    main()
