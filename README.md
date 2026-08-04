# 🛡️ Face Anti-Spoofing System

Advanced liveness detection using Vision Transformers and DINO Framework. Detects real faces vs spoofs (photos, videos, screens) in real-time.

## 🎯 Features

- ✅ Real-time face liveness detection
- ✅ 75% accuracy on custom dataset
- ✅ Detects phone screen attacks, printed photos, video replays
- ✅ Web interface with live statistics
- ✅ Trained model included (1.75 GB)
- ✅ GPU-optimized training pipeline
- ✅ Easy-to-use batch scripts

---

## 🚀 Quick Start

### 1. Web Application (Recommended)

**Easiest way to use the system:**

```bash
# Double-click or run:
run_webapp.bat

# Or manually:
python app.py
```

Then open: **http://localhost:5000**

**Features:**
- Real-time webcam detection
- Live/Spoof classification with confidence scores
- Statistics dashboard
- Image upload for analysis
- Last frame visualization after stopping

### 2. Webcam Detection (Command Line)

```bash
python test_trained_model.py --mode webcam
```

Press 'q' to quit.

### 3. Image Testing

```bash
python test_trained_model.py --mode image --input path/to/image.jpg
```

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- Webcam
- Windows/Linux/Mac

### Setup

```bash
# 1. Clone repository
git clone <your-repo-url>
cd face-antispoofing

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. For web app:
pip install -r requirements_web.txt
```

---

## 🎮 Usage

### Web Interface

```bash
# Start web server
python app.py

# Open browser
http://localhost:5000
```

**Controls:**
- **Start Detection**: Begin real-time detection
- **Stop Detection**: Stop and freeze last frame
- **Reset Stats**: Clear statistics
- **Upload Image**: Analyze a photo

### Command Line

```bash
# Webcam test
python test_trained_model.py --mode webcam

# Image test
python test_trained_model.py --mode image --input test.jpg

# Check model info
python check_model.py

# Full demo
python demo.py --model models/best_model.pth --mode webcam
```

---

## 🧪 Testing Scenarios

### Test 1: Real Face (Should be LIVE)
1. Start detection
2. Show your face to camera
3. **Expected**: GREEN box + "LIVE" label

### Test 2: Phone Screen (Should be SPOOF)
1. Start detection
2. Show a photo on your phone screen
3. **Expected**: RED box + "SPOOF" + "FAKE ALERT!"

### Test 3: Printed Photo (Should be SPOOF)
1. Start detection
2. Show a printed photo
3. **Expected**: RED box + "SPOOF"

---

## 🏋️ Training

### Using Pre-trained Model

The repository includes a trained model (`models/best_model.pth`) with 75% accuracy. You can use it directly.

### Training Your Own Model

#### Quick Training (GPU)

```bash
# Automated setup
python start_training.py

# Or manual
python train_gpu.py --dataset custom1 --epochs 30 --batch-size 32
```

#### Training Options

```bash
# Full training (50 epochs)
python train_gpu.py --dataset custom1 --epochs 50

# Fast training (30 epochs)
python train_gpu.py --dataset custom1 --epochs 30

# Memory-efficient
python train_gpu.py --dataset custom1 --batch-size 16 --accumulation-steps 2

# Resume from checkpoint
python train_gpu.py --resume models/checkpoint_epoch_20.pth --epochs 50
```

#### Expected Training Time

| Hardware | Batch Size | Time (30 epochs) |
|----------|------------|------------------|
| RTX 3060 | 32 | ~30 min |
| RTX 2060 | 16 | ~45 min |
| CPU | 8 | ~7.5 hours |

### Prepare Custom Dataset

```bash
python prepare_custom_datasets.py
```

**Dataset Structure:**
```
data/
├── custom_dataset1/
│   ├── train/
│   │   ├── live/     # Real face images
│   │   └── spoof/    # Fake/attack images
│   └── test/
│       ├── live/
│       └── spoof/
```

---

## 📊 Model Performance

- **Accuracy**: 75.27%
- **Model Size**: 1.75 GB
- **Parameters**: 218M
- **Architecture**: Vision Transformer + DINO
- **Training**: 50 epochs on custom dataset

**Performance by Class:**
- Live Detection: ~75-80%
- Spoof Detection: ~70-75%

---

## 🏗️ Project Structure

```
face-antispoofing/
├── app.py                      # Flask web application
├── config.py                   # Configuration settings
├── train_gpu.py               # GPU training script
├── demo.py                    # Demo script
├── test_trained_model.py      # Testing script
├── check_model.py             # Model info checker
├── start_training.py          # Automated training setup
├── prepare_custom_datasets.py # Dataset preparation
├── requirements.txt           # Python dependencies
├── requirements_web.txt       # Web app dependencies
├── run_webapp.bat            # Web app launcher (Windows)
├── run_model.bat             # Model launcher (Windows)
├── train.bat                 # Training launcher (Windows)
├── templates/
│   └── index.html            # Web interface
├── src/
│   ├── model_vit_dino.py     # ViT-DINO model
│   ├── dataset.py            # Dataset loaders
│   ├── preprocessing.py      # Image preprocessing
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation metrics
│   ├── inference.py          # Inference utilities
│   └── visualize.py          # Visualization tools
├── models/
│   ├── best_model.pth        # Best trained model
│   └── checkpoint_*.pth      # Training checkpoints
└── data/
    ├── custom_dataset1/      # Your dataset 1
    └── custom_dataset2/      # Your dataset 2
```

---

## 🎨 Web Interface Features

### Real-time Detection
- Live video feed with detection boxes
- Green box = LIVE (real person)
- Red box = SPOOF (fake/attack)
- Confidence scores displayed

### Statistics Dashboard
- Total detections (counts only when prediction changes)
- Live faces count
- Spoof attempts count
- Current confidence percentage

### Smart Counting
- Only increments when prediction changes
- Same face won't count multiple times
- Accurate statistics tracking

### Image Upload
- Upload any image for instant analysis
- Shows annotated result
- Displays prediction and confidence

### Visualization Persistence
- Last frame stays visible after stopping
- Can review detection results
- Shows detection boxes and labels

---

## 🔧 Configuration

### Change Detection Threshold

Edit `app.py`:
```python
# In predict_face function
if confidence < 0.7:  # 70% threshold
    label = "UNCERTAIN"
```

### Adjust Model Settings

Edit `config.py`:
```python
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
```


## 🎓 Model Architecture

### Vision Transformer (ViT)
- Patch size: 16x16
- Embedding dimension: 768
- Depth: 12 layers
- Attention heads: 12
- MLP ratio: 4.0

### DINO Framework
- Student-Teacher architecture
- Self-supervised learning
- Exponential Moving Average (EMA)
- Combined supervised + self-supervised loss


---

## 📈 Performance Tips

### For Best Detection Results
1. **Good Lighting**: Use bright, even lighting
2. **Face Position**: Look directly at camera
3. **Distance**: Keep face 30-50% of frame
4. **Camera Quality**: Use good quality webcam
5. **Stable Position**: Keep camera steady

### For Better Training
1. **More Data**: Collect diverse samples
2. **Data Augmentation**: Enable augmentation
3. **Longer Training**: Train for 50+ epochs
4. **GPU**: Use GPU for faster training
5. **Fine-tuning**: Resume from best checkpoint

---

## 🚀 Deployment

### Local Deployment
```bash
python app.py
```

### Production Deployment (Linux)
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Production Deployment (Windows)
```bash
# Install Waitress
pip install waitress

# Run with Waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```


## 🙏 Acknowledgments

- Vision Transformer (ViT) architecture
- DINO self-supervised learning framework
- PyTorch deep learning framework
- OpenCV computer vision library

---

## 🎉 Quick Commands Reference

```bash
# Web app
python app.py                                    # Start web interface
run_webapp.bat                                   # Windows launcher

# Testing
python test_trained_model.py --mode webcam      # Webcam test
python test_trained_model.py --mode image --input test.jpg  # Image test
python check_model.py                            # Check model info

# Training
python start_training.py                         # Automated training
python train_gpu.py --dataset custom1 --epochs 30  # Manual training

# Dataset
python prepare_custom_datasets.py               # Prepare dataset

# Demo
python demo.py --model models/best_model.pth --mode webcam  # Full demo
```

---

**Made with ❤️ for Face Anti-Spoofing**

🚀 **Get Started**: `python app.py` → http://localhost:5000
