# YOLO Deployment Guide

Complete guide for deploying YOLO object detection in production.

## Table of Contents

1. [Docker Deployment](#docker-deployment)
2. [REST API Deployment](#rest-api-deployment)
3. [Model Export & Optimization](#model-export--optimization)
4. [Cloud Deployment](#cloud-deployment)
5. [Edge Device Deployment](#edge-device-deployment)

## Docker Deployment

### Build Docker Image

```bash
# Build image
docker build -t yolo-detector:latest .

# Build for specific architecture
docker build --platform linux/amd64 -t yolo-detector:latest .
```

### Run Container

```bash
# CPU inference
docker run -p 5000:5000 yolo-detector:latest

# GPU inference
docker run --gpus all -p 5000:5000 yolo-detector:latest

# With volume mount for models
docker run -v $(pwd)/models:/app/models -p 5000:5000 yolo-detector:latest
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  yolo-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/models/yolov8n.pt
      - DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Run with:
```bash
docker-compose up -d
```

## REST API Deployment

### Flask API

Create `src/api/app.py`:

```python
from flask import Flask, request, jsonify
import cv2
import numpy as np
from src.detector import YOLODetector

app = Flask(__name__)
detector = YOLODetector('yolov8n.pt')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image from request
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Detect objects
    results = detector.detect(img)[0]
    
    return jsonify({
        'success': True,
        'detections': results['detections'],
        'num_detections': results['num_detections']
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Start server:
```bash
python src/api/app.py
```

Test API:
```bash
curl -X POST -F "image=@test.jpg" http://localhost:5000/predict
```

## Model Export & Optimization

### Export to ONNX

```bash
python scripts/export.py --model yolov8n.pt --format onnx

# Use exported model
python scripts/detect.py --source image.jpg --model yolov8n.onnx
```

### Export to TensorRT (NVIDIA GPUs)

```bash
# Export to TensorRT engine
python scripts/export.py --model yolov8n.pt --format engine --device 0

# Inference with TensorRT (2-3x faster)
python scripts/detect.py --source image.jpg --model yolov8n.engine
```

### Export to TFLite (Mobile/Edge)

```bash
# Export to TensorFlow Lite
python scripts/export.py --model yolov8n.pt --format tflite

# Quantize for smaller size
python scripts/export.py --model yolov8n.pt --format tflite --int8
```

### Export to CoreML (iOS)

```bash
python scripts/export.py --model yolov8n.pt --format coreml
```

## Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance

```bash
# Launch GPU instance (p3.2xlarge or g4dn.xlarge)
# Install NVIDIA drivers
sudo apt-get install nvidia-driver-525

# Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get install cuda

# Deploy with Docker
docker run --gpus all -p 80:5000 yolo-detector:latest
```

#### 2. AWS Lambda (Serverless)

```python
# lambda_function.py
import json
import boto3
from src.detector import YOLODetector

detector = YOLODetector('yolov8n.pt')

def lambda_handler(event, context):
    # Download image from S3
    s3 = boto3.client('s3')
    bucket = event['bucket']
    key = event['key']
    
    # Process image
    local_path = '/tmp/image.jpg'
    s3.download_file(bucket, key, local_path)
    
    results = detector.detect_image(local_path)
    
    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }
```

### Google Cloud Platform

```bash
# Deploy to Cloud Run
gcloud run deploy yolo-detector \
  --image gcr.io/PROJECT_ID/yolo-detector \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2
```

### Azure

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name yolo-detector \
  --image yolo-detector:latest \
  --cpu 2 \
  --memory 4 \
  --port 5000
```

## Edge Device Deployment

### NVIDIA Jetson

```bash
# Install JetPack
sudo apt-get install nvidia-jetpack

# Install PyTorch for Jetson
wget https://nvidia.box.com/shared/static/...
pip3 install torch-*.whl

# Install Ultralytics
pip3 install ultralytics

# Run detection
python3 scripts/detect.py --source 0 --model yolov8n.pt --device 0
```

### Raspberry Pi

```bash
# Use TFLite for Raspberry Pi
python scripts/export.py --model yolov8n.pt --format tflite

# Install TFLite runtime
pip install tflite-runtime

# Run inference
python scripts/detect_tflite.py --source 0 --model yolov8n.tflite
```

### Intel Neural Compute Stick

```bash
# Export to OpenVINO
python scripts/export.py --model yolov8n.pt --format openvino

# Run with OpenVINO
python scripts/detect_openvino.py --source 0 --model yolov8n_openvino_model/
```

## Performance Optimization

### 1. Model Optimization

```bash
# Use INT8 quantization (4x smaller, faster)
python scripts/export.py --model yolov8n.pt --format tflite --int8

# Use FP16 precision (2x smaller)
python scripts/export.py --model yolov8n.pt --format onnx --half
```

### 2. Batch Processing

```python
from src.detector import YOLODetector

detector = YOLODetector('yolov8n.pt')

# Process multiple images at once
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = detector.detect(image_paths)
```

### 3. Multi-GPU Training

```bash
# Use multiple GPUs for training
python scripts/train.py \
  --model yolov8n.pt \
  --data data/dataset.yaml \
  --device 0,1,2,3 \
  --batch 64
```

## Monitoring & Logging

### Add Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info('Detection started')
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

detections_counter = Counter('yolo_detections_total', 'Total detections')
inference_time = Histogram('yolo_inference_seconds', 'Inference time')

@inference_time.time()
def detect_objects(image):
    results = detector.detect(image)
    detections_counter.inc(results['num_detections'])
    return results

# Start metrics server
start_http_server(8000)
```

## Security

### 1. API Authentication

```python
from flask import request
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your-secret-key':
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ... detection code
```

### 2. Input Validation

```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def validate_image(file):
    if file.content_length > MAX_FILE_SIZE:
        raise ValueError('File too large')
    
    if not file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
        raise ValueError('Invalid file type')
```

## Scaling

### Horizontal Scaling with Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: yolo-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: yolo-detector
  template:
    metadata:
      labels:
        app: yolo-detector
    spec:
      containers:
      - name: yolo-detector
        image: yolo-detector:latest
        ports:
        - containerPort: 5000
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: yolo-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 5000
  selector:
    app: yolo-detector
```

Deploy:
```bash
kubectl apply -f deployment.yaml
```

## Troubleshooting

### Issue: Slow Inference

**Solutions:**
1. Export to TensorRT/ONNX
2. Reduce image size
3. Use smaller model (yolov8n)
4. Enable GPU
5. Batch processing

### Issue: Out of Memory

**Solutions:**
1. Reduce batch size
2. Use smaller model
3. Lower image resolution
4. Clear GPU cache: `torch.cuda.empty_cache()`

### Issue: Model Not Loading

**Solutions:**
1. Check model path
2. Ensure compatible ultralytics version
3. Re-download model weights
4. Check CUDA compatibility

---

**For more help, see [README.md](README.md) or open an issue on GitHub.**
