"""
Prepare custom datasets for training
Extracts frames from videos and organizes into train/test splits
"""

import os
import cv2
import pandas as pd
import shutil
from pathlib import Path
import random
from tqdm import tqdm


def extract_frames_from_video(video_path, output_dir, max_frames=30, skip_frames=5):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        max_frames: Maximum number of frames to extract
        skip_frames: Extract every Nth frame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return 0
    
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened() and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skip_frames == 0:
            # Save frame
            video_name = Path(video_path).stem
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    return extracted_count


def prepare_dataset1(dataset_path='dataset 1', output_path='data/custom_dataset1', 
                     train_split=0.8, frames_per_video=30):
    """
    Prepare Dataset 1 (video-based with attack types)
    
    Structure:
    - live_selfie, live_video → LIVE class
    - cut-out printouts, printouts, replay → SPOOF class
    """
    print("="*70)
    print("Preparing Dataset 1")
    print("="*70)
    
    # Create output directories
    for split in ['train', 'test']:
        for cls in ['live', 'spoof']:
            os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)
    
    # Define live and spoof categories
    live_categories = ['live_selfie', 'live_video']
    spoof_categories = ['cut-out printouts', 'printouts', 'replay']
    
    stats = {'live': 0, 'spoof': 0}
    
    # Process live samples
    print("\nProcessing LIVE samples...")
    for category in live_categories:
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            continue
        
        files = [f for f in os.listdir(category_path) 
                if f.lower().endswith(('.mp4', '.mov', '.avi', '.jpg', '.jpeg', '.png'))]
        
        random.shuffle(files)
        split_idx = int(len(files) * train_split)
        
        for idx, filename in enumerate(tqdm(files, desc=f"Processing {category}")):
            file_path = os.path.join(category_path, filename)
            split = 'train' if idx < split_idx else 'test'
            output_dir = os.path.join(output_path, split, 'live')
            
            if filename.lower().endswith(('.mp4', '.mov', '.avi')):
                # Extract frames from video
                extracted = extract_frames_from_video(file_path, output_dir, 
                                                     max_frames=frames_per_video)
                stats['live'] += extracted
            else:
                # Copy image directly
                shutil.copy(file_path, os.path.join(output_dir, filename))
                stats['live'] += 1
    
    # Process spoof samples
    print("\nProcessing SPOOF samples...")
    for category in spoof_categories:
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            continue
        
        files = [f for f in os.listdir(category_path) 
                if f.lower().endswith(('.mp4', '.mov', '.avi', '.jpg', '.jpeg', '.png'))]
        
        random.shuffle(files)
        split_idx = int(len(files) * train_split)
        
        for idx, filename in enumerate(tqdm(files, desc=f"Processing {category}")):
            file_path = os.path.join(category_path, filename)
            split = 'train' if idx < split_idx else 'test'
            output_dir = os.path.join(output_path, split, 'spoof')
            
            if filename.lower().endswith(('.mp4', '.mov', '.avi')):
                # Extract frames from video
                extracted = extract_frames_from_video(file_path, output_dir, 
                                                     max_frames=frames_per_video)
                stats['spoof'] += extracted
            else:
                # Copy image directly
                shutil.copy(file_path, os.path.join(output_dir, filename))
                stats['spoof'] += 1
    
    print("\n" + "="*70)
    print("Dataset 1 Preparation Complete!")
    print(f"Total LIVE samples: {stats['live']}")
    print(f"Total SPOOF samples: {stats['spoof']}")
    print("="*70)


def prepare_dataset2(dataset_path='dataset 2/samples', output_path='data/custom_dataset2',
                     train_split=0.8, frames_per_video=30):
    """
    Prepare Dataset 2 (user-based with live_selfie and live_video)
    
    Structure:
    - Each user folder contains live_selfie.jpg and live_video.mp4
    - All are LIVE samples (no spoof in this dataset)
    """
    print("\n" + "="*70)
    print("Preparing Dataset 2")
    print("="*70)
    
    # Create output directories
    for split in ['train', 'test']:
        os.makedirs(os.path.join(output_path, split, 'live'), exist_ok=True)
    
    # Get all user folders
    user_folders = [f for f in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, f))]
    
    random.shuffle(user_folders)
    split_idx = int(len(user_folders) * train_split)
    
    stats = {'live': 0}
    
    print(f"\nProcessing {len(user_folders)} users...")
    
    for idx, user_folder in enumerate(tqdm(user_folders, desc="Processing users")):
        user_path = os.path.join(dataset_path, user_folder)
        split = 'train' if idx < split_idx else 'test'
        output_dir = os.path.join(output_path, split, 'live')
        
        # Process live_selfie.jpg
        selfie_path = os.path.join(user_path, 'live_selfie.jpg')
        if os.path.exists(selfie_path):
            output_name = f"{user_folder}_selfie.jpg"
            shutil.copy(selfie_path, os.path.join(output_dir, output_name))
            stats['live'] += 1
        
        # Process live_video.mp4
        video_path = os.path.join(user_path, 'live_video.mp4')
        if os.path.exists(video_path):
            temp_dir = os.path.join(output_dir, 'temp')
            extracted = extract_frames_from_video(video_path, temp_dir, 
                                                 max_frames=frames_per_video)
            
            # Rename frames with user ID
            if os.path.exists(temp_dir):
                for frame_file in os.listdir(temp_dir):
                    old_path = os.path.join(temp_dir, frame_file)
                    new_name = f"{user_folder}_{frame_file}"
                    new_path = os.path.join(output_dir, new_name)
                    shutil.move(old_path, new_path)
                os.rmdir(temp_dir)
            
            stats['live'] += extracted
    
    print("\n" + "="*70)
    print("Dataset 2 Preparation Complete!")
    print(f"Total LIVE samples: {stats['live']}")
    print("Note: Dataset 2 contains only LIVE samples")
    print("="*70)


def combine_datasets(dataset1_path='data/custom_dataset1', 
                    dataset2_path='data/custom_dataset2',
                    output_path='data/combined_dataset'):
    """
    Combine both datasets into a single dataset
    """
    print("\n" + "="*70)
    print("Combining Datasets")
    print("="*70)
    
    # Create output directories
    for split in ['train', 'test']:
        for cls in ['live', 'spoof']:
            os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)
    
    stats = {'train': {'live': 0, 'spoof': 0}, 'test': {'live': 0, 'spoof': 0}}
    
    # Copy from dataset 1
    for split in ['train', 'test']:
        for cls in ['live', 'spoof']:
            src_dir = os.path.join(dataset1_path, split, cls)
            dst_dir = os.path.join(output_path, split, cls)
            
            if os.path.exists(src_dir):
                files = os.listdir(src_dir)
                for file in tqdm(files, desc=f"Copying Dataset1 {split}/{cls}"):
                    shutil.copy(os.path.join(src_dir, file), 
                              os.path.join(dst_dir, f"ds1_{file}"))
                    stats[split][cls] += 1
    
    # Copy from dataset 2 (only live samples)
    for split in ['train', 'test']:
        src_dir = os.path.join(dataset2_path, split, 'live')
        dst_dir = os.path.join(output_path, split, 'live')
        
        if os.path.exists(src_dir):
            files = os.listdir(src_dir)
            for file in tqdm(files, desc=f"Copying Dataset2 {split}/live"):
                shutil.copy(os.path.join(src_dir, file), 
                          os.path.join(dst_dir, f"ds2_{file}"))
                stats[split]['live'] += 1
    
    print("\n" + "="*70)
    print("Combined Dataset Statistics:")
    print("="*70)
    print(f"TRAIN SET:")
    print(f"  Live:  {stats['train']['live']:5d} samples")
    print(f"  Spoof: {stats['train']['spoof']:5d} samples")
    print(f"  Total: {stats['train']['live'] + stats['train']['spoof']:5d} samples")
    print(f"\nTEST SET:")
    print(f"  Live:  {stats['test']['live']:5d} samples")
    print(f"  Spoof: {stats['test']['spoof']:5d} samples")
    print(f"  Total: {stats['test']['live'] + stats['test']['spoof']:5d} samples")
    print("="*70)


def main():
    """Main function"""
    print("="*70)
    print("CUSTOM DATASET PREPARATION")
    print("="*70)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Prepare Dataset 1
    prepare_dataset1(
        dataset_path='dataset 1',
        output_path='data/custom_dataset1',
        train_split=0.8,
        frames_per_video=30
    )
    
    # Prepare Dataset 2
    prepare_dataset2(
        dataset_path='dataset 2/samples',
        output_path='data/custom_dataset2',
        train_split=0.8,
        frames_per_video=30
    )
    
    # Combine both datasets
    combine_datasets(
        dataset1_path='data/custom_dataset1',
        dataset2_path='data/custom_dataset2',
        output_path='data/combined_dataset'
    )
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print("\nYou can now train the model using:")
    print("  python src/train.py --dataset custom --epochs 100")
    print("\nDataset locations:")
    print("  - Dataset 1 only: data/custom_dataset1")
    print("  - Dataset 2 only: data/custom_dataset2")
    print("  - Combined:       data/combined_dataset")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare custom datasets')
    parser.add_argument('--frames', type=int, default=30,
                       help='Number of frames to extract per video')
    parser.add_argument('--split', type=float, default=0.8,
                       help='Train/test split ratio')
    parser.add_argument('--dataset1-only', action='store_true',
                       help='Prepare only dataset 1')
    parser.add_argument('--dataset2-only', action='store_true',
                       help='Prepare only dataset 2')
    parser.add_argument('--no-combine', action='store_true',
                       help='Do not combine datasets')
    
    args = parser.parse_args()
    
    random.seed(42)
    
    if args.dataset1_only:
        prepare_dataset1(frames_per_video=args.frames, train_split=args.split)
    elif args.dataset2_only:
        prepare_dataset2(frames_per_video=args.frames, train_split=args.split)
    else:
        main()
