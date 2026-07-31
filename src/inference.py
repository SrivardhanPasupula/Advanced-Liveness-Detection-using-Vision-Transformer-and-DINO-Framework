"""
Real-time Face Anti-Spoofing inference with webcam
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
from collections import deque
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_vit_dino import create_model
from src.preprocessing import FaceDetector, FacePreprocessor
from config import Config


class RealtimeAntispoofing:
    """Real-time face anti-spoofing detector"""
    
    def __init__(self, checkpoint_path: str, device=None, smoothing_window: int = 5):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            smoothing_window: Number of frames for temporal smoothing
        """
        self.device = device if device else Config.DEVICE
        
        # Load model
        print("Loading model...")
        self.model = create_model(Config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Model loaded successfully!")
        
        # Preprocessing
        self.face_detector = FaceDetector(method='haar')
        self.preprocessor = FacePreprocessor(image_size=Config.IMAGE_SIZE)
        
        # Temporal smoothing
        self.prediction_buffer = deque(maxlen=smoothing_window)
        self.confidence_buffer = deque(maxlen=smoothing_window)
        
        # FPS calculation
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
    
    def predict(self, image: np.ndarray) -> tuple:
        """
        Predict if face is live or spoof
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            (prediction, confidence, bbox)
            prediction: 'Live' or 'Spoof'
            confidence: Confidence score (0-1)
            bbox: Face bounding box (x, y, w, h) or None
        """
        # Detect face
        bbox = self.face_detector.detect_face(image)
        
        if bbox is None:
            return None, 0.0, None
        
        # Preprocess
        tensor = self.preprocessor.preprocess(image, bbox).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(tensor, return_features=False)
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0, 1].item()  # Probability of being live
            prediction = 1 if confidence > Config.CONFIDENCE_THRESHOLD else 0
        
        # Temporal smoothing
        self.prediction_buffer.append(prediction)
        self.confidence_buffer.append(confidence)
        
        # Average predictions
        avg_prediction = int(np.mean(self.prediction_buffer) > 0.5)
        avg_confidence = np.mean(self.confidence_buffer)
        
        label = 'Live' if avg_prediction == 1 else 'Spoof'
        
        return label, avg_confidence, bbox
    
    def draw_results(self, image: np.ndarray, prediction: str, 
                    confidence: float, bbox: tuple, fps: float = 0) -> np.ndarray:
        """
        Draw prediction results on image
        
        Args:
            image: Input image
            prediction: 'Live' or 'Spoof'
            confidence: Confidence score
            bbox: Face bounding box
            fps: Frames per second
            
        Returns:
            Annotated image
        """
        result_image = image.copy()
        
        if bbox is not None:
            x, y, w, h = bbox
            
            # Choose color based on prediction
            if prediction == 'Live':
                color = (0, 255, 0)  # Green for live
                status = "✓ LIVE"
            else:
                color = (0, 0, 255)  # Red for spoof
                status = "✗ FAKE ALERT"
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 3)
            
            # Draw status background
            status_bg_height = 80
            cv2.rectangle(result_image, (x, y - status_bg_height), 
                         (x + w, y), color, -1)
            
            # Draw status text
            cv2.putText(result_image, status, (x + 10, y - 45),
                       cv2.FONT_HERSHEY_BOLD, 0.9, (255, 255, 255), 2)
            
            # Draw confidence
            conf_text = f"Confidence: {confidence*100:.1f}%"
            cv2.putText(result_image, conf_text, (x + 10, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(result_image, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(result_image, "Press 'q' to quit", (10, result_image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image
    
    def run_webcam(self, camera_id: int = 0):
        """
        Run real-time detection on webcam
        
        Args:
            camera_id: Camera device ID
        """
        print(f"Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera opened successfully!")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time) if self.last_time else 0
            self.last_time = current_time
            self.fps_buffer.append(fps)
            avg_fps = np.mean(self.fps_buffer)
            
            # Predict
            prediction, confidence, bbox = self.predict(frame)
            
            # Draw results
            if prediction is not None:
                result_frame = self.draw_results(frame, prediction, confidence, bbox, avg_fps)
            else:
                result_frame = frame.copy()
                cv2.putText(result_frame, "No face detected", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display
            cv2.imshow('Face Anti-Spoofing - ViT + DINO', result_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed")
    
    def process_image(self, image_path: str, output_path: str = None):
        """
        Process single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save result (optional)
        """
        print(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        # Predict
        prediction, confidence, bbox = self.predict(image)
        
        # Draw results
        if prediction is not None:
            result_image = self.draw_results(image, prediction, confidence, bbox)
            print(f"Prediction: {prediction} (Confidence: {confidence*100:.2f}%)")
        else:
            result_image = image.copy()
            cv2.putText(result_image, "No face detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("No face detected")
        
        # Save or display
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
        else:
            cv2.imshow('Result', result_image)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def process_video(self, video_path: str, output_path: str = None):
        """
        Process video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save result (optional)
        """
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Predict
            prediction, confidence, bbox = self.predict(frame)
            
            # Draw results
            if prediction is not None:
                result_frame = self.draw_results(frame, prediction, confidence, bbox)
            else:
                result_frame = frame.copy()
            
            # Write or display
            if writer:
                writer.write(result_frame)
            else:
                cv2.imshow('Processing Video', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if writer:
            writer.release()
            print(f"Result saved to: {output_path}")
        cv2.destroyAllWindows()


def main(args):
    """Main inference function"""
    
    # Create detector
    detector = RealtimeAntispoofing(
        checkpoint_path=args.checkpoint,
        smoothing_window=args.smoothing
    )
    
    # Run appropriate mode
    if args.mode == 'webcam':
        detector.run_webcam(camera_id=args.camera)
    elif args.mode == 'image':
        if not args.input:
            print("Error: --input required for image mode")
            return
        detector.process_image(args.input, args.output)
    elif args.mode == 'video':
        if not args.input:
            print("Error: --input required for video mode")
            return
        detector.process_video(args.input, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time Face Anti-Spoofing Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='webcam',
                       choices=['webcam', 'image', 'video'],
                       help='Inference mode')
    parser.add_argument('--input', type=str, help='Input image/video path')
    parser.add_argument('--output', type=str, help='Output path (optional)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--smoothing', type=int, default=5,
                       help='Temporal smoothing window size')
    
    args = parser.parse_args()
    main(args)
