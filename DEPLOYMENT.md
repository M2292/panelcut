# Manga Panel Slicer - Deployment Guide

## Quick Reference - All Names & URLs

| Resource | Name/ID | Direct URL |
|----------|---------|------------|
| **Live App** | panelcut | https://panelcut-260168878394.europe-west1.run.app/ |
| **GitHub Repo** | M2292/panelcut | https://github.com/M2292/panelcut |
| **GCP Project ID** | *(your project)* | https://console.cloud.google.com/ |
| **Cloud Run Service** | `panelcut` | https://console.cloud.google.com/run/detail/europe-west1/panelcut/metrics |
| **Cloud Run Region** | `europe-west1` | - |
| **Firestore Database** | `(default)` | https://console.cloud.google.com/firestore/databases/-default-/data |
| **Firestore Collection** | `stats` | https://console.cloud.google.com/firestore/databases/-default-/data/stats |
| **Firestore Document** | `downloads` | https://console.cloud.google.com/firestore/databases/-default-/data/stats/downloads |
| **Cloud Storage Bucket** | `panelcut-training-data` | https://console.cloud.google.com/storage/browser/panelcut-training-data |
| **Training Images Path** | `obb/images/` | https://console.cloud.google.com/storage/browser/panelcut-training-data/obb/images |
| **Training Labels Path** | `obb/labels/` | https://console.cloud.google.com/storage/browser/panelcut-training-data/obb/labels |
| **Cloud Build Triggers** | - | https://console.cloud.google.com/cloud-build/triggers |
| **Cloud Build History** | - | https://console.cloud.google.com/cloud-build/builds |
| **Cloud Run Logs** | - | https://console.cloud.google.com/run/detail/europe-west1/panelcut/logs |
| **YOLO Model (GitHub)** | `manga109_yolo.pt` | https://github.com/M2292/panelcut/releases/download/v1.0.0/manga109_yolo.pt |

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Cloud Run     │────▶│    Firestore     │     │  Cloud Storage  │
│  (Flask App)    │     │  (Stats: downloads,│     │ (Training Data) │
│                 │────▶│   panels count)  │     │                 │
│  2GB Memory     │     └──────────────────┘     │ panelcut-       │
│  europe-west1   │──────────────────────────────▶│ training-data   │
└─────────────────┘                               └─────────────────┘
        │
        │ Auto-deploy on push
        ▼
┌─────────────────┐
│    GitHub       │
│  M2292/panelcut │
└─────────────────┘
```

---

## Google Cloud Services Used

| Service | Purpose | Cost |
|---------|---------|------|
| **Cloud Run** | Hosts the Flask application | Pay per request (free tier: 2M requests/month) |
| **Firestore** | Persistent download statistics | Free tier: 50K reads, 20K writes/day |
| **Cloud Storage** | Training data (images + labels) | ~$0.02/GB/month |
| **Cloud Build** | Auto-builds Docker image on git push | Free tier: 120 build-minutes/day |

---

## Deployment Setup (Already Done)

### 1. Cloud Run Configuration
- **Region**: europe-west1
- **Memory**: 2 GiB (required for YOLO model)
- **CPU**: 1
- **Min instances**: 0 (scales to zero when idle)
- **Max instances**: 10
- **Authentication**: Allow unauthenticated (public access)
- **Trigger**: Automatic deploy on push to `main` branch

### 2. Firestore Database
- **Mode**: Native mode
- **Region**: europe-west1
- **Collection**: `stats`
- **Document**: `downloads` containing `{total_downloads, total_panels}`

### 3. Cloud Storage Bucket
- **Name**: `panelcut-training-data`
- **Region**: europe-west1
- **Structure**:
  ```
  panelcut-training-data/
  └── obb/
      ├── images/
      │   ├── manga_20240115_abc123.png
      │   └── ...
      └── labels/
          ├── manga_20240115_abc123.txt
          └── ...
  ```

---

## Training Data Collection

Training data is collected **silently** when users download panels. The app saves:
1. Original manga page image (PNG)
2. OBB label file with panel coordinates (TXT)

### OBB Label Format
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```
- `class_id`: Always 0 (panel class)
- `x1,y1` to `x4,y4`: Four corners of the bounding box (normalized 0-1)

Example:
```
0 0.100000 0.100000 0.400000 0.100000 0.400000 0.500000 0.100000 0.500000
0 0.500000 0.100000 0.900000 0.100000 0.900000 0.600000 0.500000 0.600000
```

---

## Downloading Training Data

### Option 1: Google Cloud Console (GUI)
1. Go to https://console.cloud.google.com/storage/browser/panelcut-training-data
2. Navigate to `obb/images/` and `obb/labels/`
3. Select files and click "Download"

### Option 2: gsutil Command Line
```bash
# Install Google Cloud SDK first: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Download all training data
gsutil -m cp -r gs://panelcut-training-data/obb ./training_data/

# Download only images
gsutil -m cp -r gs://panelcut-training-data/obb/images ./training_data/obb/

# Download only labels
gsutil -m cp -r gs://panelcut-training-data/obb/labels ./training_data/obb/
```

### Option 3: Python Script
```python
from google.cloud import storage

def download_training_data(destination_folder='./training_data'):
    client = storage.Client()
    bucket = client.bucket('panelcut-training-data')

    blobs = bucket.list_blobs(prefix='obb/')
    for blob in blobs:
        local_path = f"{destination_folder}/{blob.name}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded: {blob.name}")

download_training_data()
```

---

## Training a New OBB Model

Once you have enough training data (50+ samples recommended):

```python
from ultralytics import YOLO

# Create dataset.yaml
dataset_config = """
path: ./training_data/obb
train: images
val: images
names:
  0: panel
"""

with open('./training_data/obb/dataset.yaml', 'w') as f:
    f.write(dataset_config)

# Train OBB model
model = YOLO('yolov8n-obb.pt')  # Start from pretrained OBB model
results = model.train(
    data='./training_data/obb/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# Best model saved to: runs/obb/train/weights/best.pt
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | - | Set to `production` in Cloud Run |
| `MODEL_URL` | - | URL to download YOLO model during build |
| `TRAINING_BUCKET` | `panelcut-training-data` | Cloud Storage bucket name |
| `MAX_UPLOAD_SIZE` | `15728640` (15MB) | Max upload file size |
| `SECRET_KEY` | `dev-key-...` | Flask secret key |

---

## Monitoring

### View Logs
```bash
gcloud run logs read --service panelcut --region europe-west1
```

Or in console: https://console.cloud.google.com/run/detail/europe-west1/panelcut/logs

### Check Stats
Visit: https://panelcut-260168878394.europe-west1.run.app/api/stats

### View Firestore Data
https://console.cloud.google.com/firestore/databases/-default-/data/stats/downloads

---

## Redeploying

Deployments happen automatically when you push to `main`:

```bash
git add .
git commit -m "Your changes"
git push
```

Cloud Build will:
1. Detect the push
2. Build Docker image
3. Deploy to Cloud Run
4. Route traffic to new revision

Build status: https://console.cloud.google.com/cloud-build/builds

---

## Costs Estimate

For low-moderate usage (< 1000 users/month):

| Service | Estimated Cost |
|---------|---------------|
| Cloud Run | $0 (within free tier) |
| Firestore | $0 (within free tier) |
| Cloud Storage | $0.02-0.10/month |
| Cloud Build | $0 (within free tier) |
| **Total** | **~$0.10/month** |

---

## Troubleshooting

### Cold Start Slowness
First request after idle period takes 10-30 seconds (loading YOLO model). This is normal for scale-to-zero.

### Build Failures
Check Cloud Build logs: https://console.cloud.google.com/cloud-build/builds

### Stats Not Persisting
Verify Firestore is accessible:
```bash
curl https://panelcut-260168878394.europe-west1.run.app/api/stats
```

### Training Data Not Saving
Check Cloud Run logs for `[Training]` messages:
```bash
gcloud run logs read --service panelcut --region europe-west1 | grep Training
```
