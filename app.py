"""
Flask Web Application for Face Anti-Spoofing
Real-time webcam detection with trained model
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import torch
import numpy as np
from src.model_vit_dino import create_model
from src.preprocessing import FacePreprocessor
from config import Config
import base64
from datetime import datetime
import os

app = Flask(__name__)

# Global variables
model = None
preprocessor = None
face_cascade = None
camera = None
detection_active = False
last_frame = None  # Store last frame for visualization
stats = {
    'total_detections': 0,
    'live_count': 0,
    'spoof_count': 0,
    'last_prediction': None,
    'last_confidence': 0,
    'model_accuracy': 0,
    'previous_prediction': None  # Track previous prediction to avoid duplicate counts
}

def load_model():
    """Load the trained model"""
    global model, preprocessor, face_cascade, stats
    
    print("Loading model...")
    
    # Create model
    model = create_model(Config)
    
    # Load trained weights
    model_path = 'models/best_model.pth'
    checkpoint = torch.load(model_path, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    model.eval()
    
    # Store model accuracy
    stats['model_accuracy'] = checkpoint.get('accuracy', 0)
    
    # Preprocessor
    preprocessor = FacePreprocessor()
    
    # Face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    print(f"✓ Model loaded (Accuracy: {stats['model_accuracy']:.2f}%)")

def predict_face(image):
    """Predict if face is live or spoof"""
    global model, preprocessor
    
    # Preprocess
    img_tensor = preprocessor.preprocess(image)
    img_tensor = img_tensor.unsqueeze(0).to(Config.DEVICE)
    
    # Predict
    with torch.no_grad():
        logits = model(img_tensor, return_features=False)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    
    # 0 = spoof, 1 = live
    label = "LIVE" if pred == 1 else "SPOOF"
    
    return label, confidence

def generate_frames():
    """Generate video frames with detection"""
    global camera, detection_active, stats, last_frame
    
    camera = cv2.VideoCapture(0)
    
    while detection_active:
        success, frame = camera.read()
        if not success:
            break
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face
            face = frame[y:y+h, x:x+w]
            
            # Predict
            label, confidence = predict_face(face)
            
            # Update stats only if prediction changed
            if label != stats['previous_prediction']:
                stats['total_detections'] += 1
                
                if label == "LIVE":
                    stats['live_count'] += 1
                else:
                    stats['spoof_count'] += 1
                
                stats['previous_prediction'] = label
            
            # Always update current prediction and confidence
            stats['last_prediction'] = label
            stats['last_confidence'] = confidence
            
            # Draw results
            color = (0, 255, 0) if label == "LIVE" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Text background
            text = f"{label}: {confidence*100:.1f}%"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x, y-35), (x+text_w, y), color, -1)
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Alert for spoof
            if label == "SPOOF":
                cv2.rectangle(frame, (10, 10), (300, 70), (0, 0, 255), -1)
                cv2.putText(frame, "FAKE ALERT!", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Store last frame
        last_frame = frame.copy()
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    # Send last frame one more time when stopping
    if last_frame is not None:
        ret, buffer = cv2.imencode('.jpg', last_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    if camera:
        camera.release()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start detection"""
    global detection_active
    detection_active = True
    return jsonify({'status': 'started'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop detection"""
    global detection_active, camera, last_frame
    detection_active = False
    if camera:
        camera.release()
    
    # Save last frame as static image
    if last_frame is not None:
        _, buffer = cv2.imencode('.jpg', last_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            'status': 'stopped',
            'last_frame': f'data:image/jpeg;base64,{img_base64}'
        })
    
    return jsonify({'status': 'stopped'})

@app.route('/stats')
def get_stats():
    """Get detection statistics"""
    return jsonify(stats)

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    """Reset statistics"""
    global stats
    stats['total_detections'] = 0
    stats['live_count'] = 0
    stats['spoof_count'] = 0
    stats['previous_prediction'] = None
    return jsonify({'status': 'reset'})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Upload and analyze image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    # Read image
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect face
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return jsonify({'error': 'No face detected'}), 400
    
    # Predict first face
    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    label, confidence = predict_face(face)
    
    # Draw on image
    color = (0, 255, 0) if label == "LIVE" else (0, 0, 255)
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
    text = f"{label}: {confidence*100:.1f}%"
    cv2.putText(image, text, (x, y-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Encode result
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'prediction': label,
        'confidence': float(confidence),
        'image': f'data:image/jpeg;base64,{img_base64}'
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run app
    print("\n" + "="*60)
    print("🚀 Face Anti-Spoofing Web Application")
    print("="*60)
    print(f"Model Accuracy: {stats['model_accuracy']:.2f}%")
    print(f"Device: {Config.DEVICE}")
    print("\n🌐 Open in browser: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
