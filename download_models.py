"""
Auto-download YOLO models for manga panel detection.
This runs on server startup if models are missing.
"""

import os
import urllib.request
import sys

# Model configurations
MODELS = {
    'manga109_yolo': {
        # Using a publicly available manga panel detection model
        # You can replace this URL with your own hosted model
        'url': os.environ.get('MODEL_URL', ''),
        'path': 'models/manga109_yolo.pt',
        'required': True
    }
}

def download_file(url: str, destination: str, description: str = "file") -> bool:
    """Download a file with progress indication."""
    if not url:
        print(f"[Models] No URL configured for {description}")
        return False

    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        print(f"[Models] Downloading {description}...")
        print(f"[Models] URL: {url}")
        print(f"[Models] Destination: {destination}")

        # Download with progress
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r[Models] Progress: {percent}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, destination, progress_hook)
        print(f"\n[Models] Successfully downloaded {description}")
        return True

    except Exception as e:
        print(f"\n[Models] Failed to download {description}: {e}")
        return False


def ensure_models():
    """Ensure all required models are downloaded."""
    print("[Models] Checking for required models...")

    for name, config in MODELS.items():
        path = config['path']

        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"[Models] {name}: Found ({size_mb:.1f} MB)")
        else:
            print(f"[Models] {name}: Not found at {path}")

            if config.get('url'):
                success = download_file(
                    config['url'],
                    path,
                    description=name
                )
                if not success and config.get('required'):
                    print(f"[Models] WARNING: Required model {name} could not be downloaded")
            else:
                print(f"[Models] No download URL configured for {name}")
                if config.get('required'):
                    print(f"[Models] Set MODEL_URL environment variable to auto-download")


if __name__ == '__main__':
    ensure_models()
