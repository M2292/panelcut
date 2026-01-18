# YOLO Models for Manga Panel Detection

Place your trained YOLO models here (.pt files).

## Training Your Own Model

### Option 1: Fine-tune on Manga109 Dataset

1. Download Manga109 dataset from http://www.manga109.org/en/
2. Convert annotations to YOLO format
3. Train:

```python
from ultralytics import YOLO

# Load base model
model = YOLO('yolov8n.pt')  # or yolov8s.pt for better accuracy

# Train on your dataset
model.train(
    data='manga_panels.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='manga_panels'
)

# Export trained model
# Model will be saved to runs/detect/manga_panels/weights/best.pt
```

### Option 2: Use Pre-trained Models

Community models may be available at:
- Hugging Face: https://huggingface.co/models?search=manga
- Roboflow Universe: https://universe.roboflow.com/

### YAML Configuration Example (manga_panels.yaml)

```yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: panel
  1: text
  2: balloon
```

## Recommended Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolov8n | 6MB | Fastest | Good | Quick testing |
| yolov8s | 22MB | Fast | Better | Balanced |
| yolov8m | 52MB | Medium | Best | Production |
