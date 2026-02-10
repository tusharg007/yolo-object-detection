# YOLO Object Detection - Quick Start Guide

Get started with YOLO object detection in minutes!

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/tusharg007/yolo-object-detection.git
cd yolo-object-detection
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install ultralytics YOLO
pip install ultralytics
```

## Basic Usage

### Detect Objects in Images

```bash
# Download a pre-trained model (first time only)
# This will be done automatically

# Detect objects in an image
python scripts/detect.py --source path/to/image.jpg --model yolov8n.pt

# Save results
python scripts/detect.py --source image.jpg --model yolov8n.pt --save

# Show real-time results
python scripts/detect.py --source image.jpg --model yolov8n.pt --show
```

### Detect Objects in Videos

```bash
# Process video file
python scripts/detect.py --source video.mp4 --model yolov8n.pt --save

# Real-time webcam detection
python scripts/detect.py --source 0 --model yolov8n.pt --show
```

### Train Custom Model

#### Step 1: Prepare Dataset

Organize your data:
```
data/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       ├── img3.jpg
│       └── img4.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        ├── img3.txt
        └── img4.txt
```

Label format (YOLO format - one object per line):
```
class_id center_x center_y width height
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.2
```

All coordinates are normalized (0-1).

#### Step 2: Update Dataset Config

Edit `data/dataset.yaml`:
```yaml
path: ./data
train: images/train
val: images/val

nc: 3  # number of classes
names: ['class1', 'class2', 'class3']  # your class names
```

#### Step 3: Train

```bash
# Train with default config
python scripts/train.py --config configs/train_config.yaml

# Or specify parameters directly
python scripts/train.py \
    --model yolov8n.pt \
    --data data/dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640
```

## Model Variants

Choose the right model for your needs:

| Model | Size | mAP | Speed (FPS) | Use Case |
|-------|------|-----|-------------|----------|
| yolov8n | 6 MB | 37.3 | 45 | Mobile/Edge devices |
| yolov8s | 22 MB | 44.9 | 35 | Balanced performance |
| yolov8m | 52 MB | 50.2 | 28 | High accuracy |
| yolov8l | 87 MB | 52.9 | 22 | Production systems |
| yolov8x | 136 MB | 53.9 | 18 | Maximum accuracy |

## Python API Usage

```python
from src.detector import YOLODetector

# Initialize detector
detector = YOLODetector('yolov8n.pt')

# Detect in image
results = detector.detect_image('image.jpg', save_path='output.jpg')
print(f"Found {results['num_detections']} objects")

# Print detections
for det in results['detections']:
    print(f"{det['class_name']}: {det['confidence']:.2f}")

# Detect in video
results = detector.detect_video('video.mp4', output_path='output.mp4')

# Real-time webcam
results = detector.detect_video(0, show=True)  # 0 = webcam

# Benchmark performance
metrics = detector.benchmark()
print(f"FPS: {metrics['fps']:.2f}")
```

## Configuration

### Detection Settings

Edit `configs/detect_config.yaml`:

```yaml
# Confidence threshold (0-1)
conf_threshold: 0.25

# IoU threshold for NMS (0-1)  
iou_threshold: 0.45

# Maximum detections per image
max_detections: 300

# Filter specific classes (null = all classes)
classes: null  # or [0, 1, 2] for specific classes
```

### Training Settings

Edit `configs/train_config.yaml`:

```yaml
# Model
model: yolov8n.pt

# Dataset
data: data/dataset.yaml

# Training
epochs: 100
batch: 16
imgsz: 640
lr0: 0.01

# Device
device: 0  # GPU id or 'cpu'
```

## Common Use Cases

### 1. Security Camera Monitoring

```bash
# Detect people in real-time
python scripts/detect.py \
    --source 0 \
    --model yolov8n.pt \
    --classes 0 \
    --conf 0.5 \
    --save \
    --show
```

### 2. Vehicle Detection

```bash
# Detect cars, trucks, buses
python scripts/detect.py \
    --source traffic.mp4 \
    --model yolov8n.pt \
    --classes 2 5 7 \
    --save
```

### 3. Wildlife Monitoring

```bash
# Train on custom wildlife dataset
python scripts/train.py \
    --model yolov8s.pt \
    --data wildlife_dataset.yaml \
    --epochs 200 \
    --imgsz 1280
```

## Performance Optimization

### 1. Use Appropriate Model Size

- **Mobile/Edge**: yolov8n
- **Balanced**: yolov8s or yolov8m
- **High accuracy**: yolov8l or yolov8x

### 2. Adjust Image Size

```bash
# Faster inference (lower accuracy)
python scripts/detect.py --source image.jpg --imgsz 320

# Higher accuracy (slower)
python scripts/detect.py --source image.jpg --imgsz 1280
```

### 3. Use GPU

```bash
python scripts/detect.py --source image.jpg --device 0  # GPU 0
```

### 4. Export to ONNX (faster inference)

```bash
python scripts/export.py --model yolov8n.pt --format onnx

# Use exported model
python scripts/detect.py --source image.jpg --model yolov8n.onnx
```

## Troubleshooting

### Issue: Out of Memory

**Solution:**
```bash
# Reduce batch size
python scripts/train.py --batch 8

# Or use smaller model
python scripts/train.py --model yolov8n.pt
```

### Issue: Low FPS

**Solution:**
```bash
# Use smaller model
python scripts/detect.py --model yolov8n.pt

# Reduce image size
python scripts/detect.py --imgsz 320

# Use GPU
python scripts/detect.py --device 0
```

### Issue: Poor Detection Accuracy

**Solution:**
1. Increase confidence threshold: `--conf 0.5`
2. Use larger model: `yolov8m.pt` or `yolov8l.pt`
3. Train on your custom dataset
4. Increase training epochs

## Next Steps

1. **Try different models**: Test yolov8s, yolov8m for better accuracy
2. **Train custom model**: Use your own dataset
3. **Optimize for deployment**: Export to ONNX or TensorRT
4. **Build API**: Use Flask/FastAPI for web deployment
5. **Add tracking**: Implement object tracking across frames

## Resources

- **Documentation**: See `docs/` directory
- **Examples**: See `notebooks/` directory
- **Issues**: Open an issue on GitHub
- **YOLO Docs**: https://docs.ultralytics.com/

## Quick Commands Reference

```bash
# Detection
python scripts/detect.py --source IMAGE --model MODEL

# Training
python scripts/train.py --config CONFIG

# Validation
python scripts/validate.py --model MODEL --data DATA

# Export
python scripts/export.py --model MODEL --format FORMAT

# Benchmark
python scripts/benchmark.py --model MODEL
```

---

**Happy Detecting! 🎯**

For detailed documentation, see [README.md](README.md)
