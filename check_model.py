"""
Check trained model information
"""

import torch
import os

def check_model(model_path):
    """Check model information"""
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Checking: {model_path}")
    print(f"{'='*60}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Display information
    print(f"\n📊 Model Information:")
    print(f"  File size: {os.path.getsize(model_path) / 1e6:.2f} MB")
    
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    
    if 'accuracy' in checkpoint:
        print(f"  Accuracy: {checkpoint['accuracy']:.2f}%")
    
    if 'live_acc' in checkpoint:
        print(f"  Live Accuracy: {checkpoint['live_acc']:.2f}%")
    
    if 'spoof_acc' in checkpoint:
        print(f"  Spoof Accuracy: {checkpoint['spoof_acc']:.2f}%")
    
    # Check model state
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"  Parameters: {total_params:,}")
    
    print()

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🔍 Trained Model Checker")
    print("="*60)
    
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        print(f"\n❌ Models directory not found: {models_dir}")
        return
    
    # List all models
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print(f"\n❌ No model files found in {models_dir}")
        return
    
    print(f"\n✓ Found {len(model_files)} model(s):")
    for i, model_file in enumerate(model_files, 1):
        print(f"  {i}. {model_file}")
    
    # Check best model first
    best_model_path = os.path.join(models_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        check_model(best_model_path)
    
    # Check all checkpoints
    checkpoints = sorted([f for f in model_files if f.startswith('checkpoint_')])
    
    if checkpoints:
        print(f"\n{'='*60}")
        print("📈 Training Progress (Checkpoints)")
        print(f"{'='*60}\n")
        
        for checkpoint_file in checkpoints:
            checkpoint_path = os.path.join(models_dir, checkpoint_file)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            epoch = checkpoint.get('epoch', '?')
            acc = checkpoint.get('accuracy', 0)
            
            print(f"  Epoch {epoch:2d}: {acc:.2f}% accuracy")
    
    # Summary
    print(f"\n{'='*60}")
    print("✅ Model Check Complete")
    print(f"{'='*60}")
    print("\n💡 To use your trained model:")
    print("   python demo.py --model models/best_model.pth --mode webcam")
    print("\n💡 To evaluate your model:")
    print("   python src/evaluate.py --model models/best_model.pth --dataset custom1")
    print()

if __name__ == '__main__':
    main()
