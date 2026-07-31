"""
Quick Start Training Script
Checks system, GPU, and starts training with optimal settings
"""

import os
import sys
import torch
import platform
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text)
    print("="*60)

def check_gpu():
    """Check GPU availability and details"""
    print_header("🔍 Checking GPU")
    
    if torch.cuda.is_available():
        print(f"✓ GPU Available: YES")
        print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ PyTorch Version: {torch.__version__}")
        return True
    else:
        print("❌ GPU Available: NO")
        print("⚠ Training will use CPU (very slow)")
        return False

def check_data():
    """Check if datasets exist"""
    print_header("📁 Checking Datasets")
    
    datasets = {
        'custom1': 'data/custom_dataset1',
        'custom2': 'data/custom_dataset2',
        'combined': 'data/combined_dataset'
    }
    
    available = []
    
    for name, path in datasets.items():
        train_path = os.path.join(path, 'train')
        test_path = os.path.join(path, 'test')
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            # Count samples
            live_train = len([f for f in os.listdir(os.path.join(train_path, 'live')) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(os.path.join(train_path, 'live')) else 0
            spoof_train = len([f for f in os.listdir(os.path.join(train_path, 'spoof')) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(os.path.join(train_path, 'spoof')) else 0
            live_test = len([f for f in os.listdir(os.path.join(test_path, 'live')) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(os.path.join(test_path, 'live')) else 0
            spoof_test = len([f for f in os.listdir(os.path.join(test_path, 'spoof')) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(os.path.join(test_path, 'spoof')) else 0
            
            total_train = live_train + spoof_train
            total_test = live_test + spoof_test
            
            if total_train > 0 and total_test > 0:
                print(f"✓ {name}: {total_train} train, {total_test} test samples")
                available.append(name)
            else:
                print(f"❌ {name}: Empty dataset")
        else:
            print(f"❌ {name}: Not found")
    
    return available

def recommend_settings(has_gpu, gpu_memory=None):
    """Recommend training settings based on hardware"""
    print_header("⚙️ Recommended Settings")
    
    if has_gpu:
        if gpu_memory and gpu_memory >= 12:
            batch_size = 64
            epochs = 50
            workers = 4
            print("✓ High-end GPU detected")
        elif gpu_memory and gpu_memory >= 8:
            batch_size = 32
            epochs = 50
            workers = 4
            print("✓ Mid-range GPU detected")
        else:
            batch_size = 16
            epochs = 30
            workers = 2
            print("✓ Entry-level GPU detected")
        
        print(f"  Batch Size: {batch_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Workers: {workers}")
        print(f"  Mixed Precision: Enabled")
        print(f"  Estimated Time: {epochs * 60 / 60:.1f} - {epochs * 90 / 60:.1f} minutes")
        
        return batch_size, epochs, workers
    else:
        print("⚠ CPU Training (Not Recommended)")
        print("  Batch Size: 8")
        print("  Epochs: 10 (for testing only)")
        print("  Workers: 0")
        print("  Estimated Time: 3-5 hours")
        print("\n💡 Recommendation: Use Google Colab for free GPU")
        return 8, 10, 0

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🚀 Face Anti-Spoofing Training Setup")
    print("="*60)
    
    # Check system
    print(f"\n💻 System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Check GPU
    has_gpu = check_gpu()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if has_gpu else None
    
    # Check datasets
    available_datasets = check_data()
    
    if not available_datasets:
        print("\n❌ No datasets found!")
        print("\n📝 Please prepare your dataset:")
        print("  1. Run: python prepare_custom_datasets.py")
        print("  2. Or check: CUSTOM_DATASET_GUIDE.md")
        return
    
    # Recommend settings
    batch_size, epochs, workers = recommend_settings(has_gpu, gpu_memory)
    
    # Choose dataset
    print_header("📊 Select Dataset")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"  {i}. {dataset}")
    
    try:
        choice = input(f"\nEnter choice (1-{len(available_datasets)}) or press Enter for '{available_datasets[0]}': ").strip()
        if choice == '':
            dataset = available_datasets[0]
        else:
            dataset = available_datasets[int(choice) - 1]
    except (ValueError, IndexError):
        dataset = available_datasets[0]
    
    print(f"\n✓ Selected dataset: {dataset}")
    
    # Confirm training
    print_header("🎯 Ready to Train")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {'GPU' if has_gpu else 'CPU'}")
    
    confirm = input("\nStart training? (Y/n): ").strip().lower()
    
    if confirm in ['', 'y', 'yes']:
        print_header("🚀 Starting Training")
        
        # Build command
        if has_gpu:
            cmd = [
                sys.executable, 'train_gpu.py',
                '--dataset', dataset,
                '--epochs', str(epochs),
                '--batch-size', str(batch_size),
                '--num-workers', str(workers)
            ]
        else:
            cmd = [
                sys.executable, 'src/train.py',
                '--dataset', dataset,
                '--epochs', str(epochs),
                '--batch-size', str(batch_size)
            ]
        
        print(f"\nCommand: {' '.join(cmd)}\n")
        
        # Run training
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n\n⚠ Training interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
    else:
        print("\n❌ Training cancelled")
        print("\n💡 To train manually:")
        if has_gpu:
            print(f"   python train_gpu.py --dataset {dataset} --epochs {epochs} --batch-size {batch_size}")
        else:
            print(f"   python src/train.py --dataset {dataset} --epochs {epochs}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Setup cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
