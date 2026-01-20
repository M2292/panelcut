"""
Download Training Data from Google Cloud Storage

This script downloads both OBB and Seg v2 training data from your GCS bucket
and merges it with your local training data.
"""

import os
import sys
from pathlib import Path

def download_from_gcs():
    """Download training data from Google Cloud Storage bucket."""
    try:
        from google.cloud import storage
    except ImportError:
        print("[ERROR] Google Cloud Storage library not installed!")
        print("\nInstall with:")
        print("  pip install google-cloud-storage")
        return False

    BUCKET_NAME = 'panelcut-training-data'
    MODEL_VERSIONS = ['obb', 'seg_v2']  # Download both OBB and Seg v2 data

    print("=" * 70)
    print("  Download Training Data from Google Cloud Storage")
    print("=" * 70)
    print(f"\nBucket: {BUCKET_NAME}")
    print(f"Model versions: {', '.join(MODEL_VERSIONS)}")
    print()

    try:
        # Initialize GCS client
        print("Connecting to Google Cloud Storage...")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # List all blobs in the bucket
        print("Fetching file list from cloud...")
        blobs = list(bucket.list_blobs())

        if not blobs:
            print("\n[WARNING] No files found in the bucket!")
            return True

        print(f"Found {len(blobs)} files in cloud storage")
        print()

        # Download files for each model version
        total_downloaded = 0
        total_skipped = 0
        total_errors = 0

        for model_version in MODEL_VERSIONS:
            print(f"\n--- Processing {model_version.upper()} ---")

            # Create local directories for this model version
            local_dir = Path('training_data') / model_version
            images_dir = local_dir / 'images'
            labels_dir = local_dir / 'labels'
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Filter blobs for this model version
            version_blobs = [b for b in blobs if b.name.startswith(f"{model_version}/")]

            if not version_blobs:
                print(f"  No {model_version} files found in cloud")
                continue

            print(f"  Found {len(version_blobs)} {model_version} files")

            downloaded = 0
            skipped = 0
            errors = 0

            for blob in version_blobs:
                try:
                    # Get filename without the model_version prefix
                    # blob.name is like "obb/images/file.png" or "seg_v2/labels/file.txt"
                    parts = blob.name.split('/')
                    if len(parts) < 3:  # Skip directory markers
                        continue

                    filename = parts[2]  # The actual filename

                    if not filename:
                        continue

                    # Determine destination directory based on the middle part
                    if 'images' in parts[1]:
                        local_path = images_dir / filename
                    elif 'labels' in parts[1]:
                        local_path = labels_dir / filename
                    else:
                        skipped += 1
                        continue

                    # Skip if already exists
                    if local_path.exists():
                        skipped += 1
                        continue

                    # Download
                    blob.download_to_filename(str(local_path))
                    downloaded += 1

                    if downloaded % 10 == 0:
                        print(f"    Downloaded {downloaded} files...")

                except Exception as e:
                    print(f"    ERROR downloading {blob.name}: {e}")
                    errors += 1

            print(f"  {model_version}: {downloaded} downloaded, {skipped} skipped, {errors} errors")
            total_downloaded += downloaded
            total_skipped += skipped
            total_errors += errors

        print("\n" + "=" * 70)
        print("Download Complete!")
        print("=" * 70)
        print(f"Total files downloaded: {total_downloaded}")
        print(f"Total files skipped (already exist): {total_skipped}")
        print(f"Total errors: {total_errors}")
        print()

        # Count total local files for each version
        for model_version in MODEL_VERSIONS:
            local_dir = Path('training_data') / model_version
            images_dir = local_dir / 'images'
            labels_dir = local_dir / 'labels'

            if images_dir.exists() and labels_dir.exists():
                total_images = len(list(images_dir.glob('*.png'))) + len(list(images_dir.glob('*.jpg')))
                total_labels = len(list(labels_dir.glob('*.txt')))

                print(f"{model_version.upper()} training data:")
                print(f"  Images: {total_images}")
                print(f"  Labels: {total_labels}")

                if total_images != total_labels and total_images > 0:
                    print(f"  [WARNING] Mismatch between images and labels!")
                print()

        print("=" * 70)
        return True

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Failed to download from GCS:")
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("\n")

    # Auto-detect and set credentials from local key file
    script_dir = Path(__file__).parent
    key_file = script_dir / 'panelcut-bd9b32e860e1.json'

    if key_file.exists() and 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
        print(f"Found service account key: {key_file.name}")
        print("Using local credentials for authentication...")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(key_file)
        print()
    elif 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
        print("[WARNING] GOOGLE_APPLICATION_CREDENTIALS not set!")
        print()
        print("To authenticate with Google Cloud:")
        print("1. Download service account key JSON from GCP Console")
        print("2. Save it as 'panelcut-bd9b32e860e1.json' in this folder")
        print("   OR set environment variable:")
        print("   set GOOGLE_APPLICATION_CREDENTIALS=C:\\path\\to\\key.json")
        print()
        print("Or run: gcloud auth application-default login")
        print()

        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    success = download_from_gcs()

    if success:
        print("\n✓ Training data successfully downloaded!")
        print()
        print("Next steps:")
        print("  1. Review the data:")
        print("     - OBB data: training_data/obb/")
        print("     - Seg v2 data: training_data/seg_v2/")
        print("  2. Train models:")
        print("     - OBB: train_obb_model.bat")
        print("     - Seg v2: train_seg_v2_model.bat")
        print("  3. Models will use combined local + cloud data")
    else:
        print("\n✗ Download failed - check errors above")

    input("\nPress Enter to exit...")


if __name__ == '__main__':
    main()
