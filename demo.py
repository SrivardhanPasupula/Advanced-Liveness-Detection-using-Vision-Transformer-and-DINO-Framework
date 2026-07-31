"""
Quick demo script for Face Anti-Spoofing
"""

import sys
import os

sys.path.append('src')

from src.inference import RealtimeAntispoofing


def run_demo(checkpoint_path='models/best_model.pth'):
    """
    Run a quick demo of the face anti-spoofing system
    
    Args:
        checkpoint_path: Path to trained model checkpoint
    """
    
    print("="*60)
    print("Face Anti-Spoofing Demo")
    print("ViT + DINO Framework")
    print("="*60)
    
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Model checkpoint not found at {checkpoint_path}")
        print("\nPlease train the model first:")
        print("  python src/train.py --dataset casia --epochs 100")
        return
    
    # Create detector
    print("\nInitializing detector...")
    detector = RealtimeAntispoofing(checkpoint_path)
    
    print("\nDemo Options:")
    print("1. Webcam (real-time)")
    print("2. Image file")
    print("3. Video file")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        print("\nStarting webcam detection...")
        print("Press 'q' to quit")
        detector.run_webcam()
    
    elif choice == '2':
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            detector.process_image(image_path)
        else:
            print(f"Error: Image not found at {image_path}")
    
    elif choice == '3':
        video_path = input("Enter video path: ").strip()
        if os.path.exists(video_path):
            save = input("Save output? (y/n): ").strip().lower()
            output_path = 'output_video.mp4' if save == 'y' else None
            detector.process_video(video_path, output_path)
        else:
            print(f"Error: Video not found at {video_path}")
    
    else:
        print("Invalid option")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Anti-Spoofing Demo')
    parser.add_argument('--checkpoint', type=str, default='models/best_model.pth',
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    run_demo(args.checkpoint)
