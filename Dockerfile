FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download model during build (hardcoded URL for Cloud Run)
ENV MODEL_URL=https://github.com/M2292/panelcut/releases/download/v1.0.0/manga109_yolo.pt
RUN python download_models.py

# Create necessary directories
RUN mkdir -p uploads output training_data/obb/images training_data/obb/labels

# Set environment variables
ENV FLASK_ENV=production
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run with gunicorn (single worker to save memory)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 120 app:app
