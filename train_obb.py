"""
Train OBB (Oriented Bounding Box) Model

This script trains a YOLOv8-OBB model on collected training data.
Run via train_obb_model.bat or directly: python train_obb.py
"""

import os
import sys
import random
import shutil

sys.path.insert(0, '.')

def main():
    try:
        from ultralytics import YOLO
        import yaml
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("\nInstall with: pip install ultralytics pyyaml")
        return False

    TRAINING_DATA_FOLDER = 'training_data'
    obb_data_folder = os.path.join(TRAINING_DATA_FOLDER, 'obb')
    obb_images_dir = os.path.join(obb_data_folder, 'images')
    obb_labels_dir = os.path.join(obb_data_folder, 'labels')

    # Check if data exists
    if not os.path.exists(obb_images_dir):
        print(f"[ERROR] Training images directory not found: {obb_images_dir}")
        return False

    # Get image files
    image_files = [f for f in os.listdir(obb_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f'Found {len(image_files)} training images')

    if len(image_files) < 5:
        print(f"[ERROR] Need at least 5 training images, found {len(image_files)}")
        return False

    # Split 80/20
    random.shuffle(image_files)
    split_idx = max(1, int(len(image_files) * 0.8))
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:] if split_idx < len(image_files) else [image_files[-1]]

    print(f'Train: {len(train_images)}, Val: {len(val_images)}')

    # Create train/val dirs
    train_img_dir = os.path.join(obb_data_folder, 'train', 'images')
    train_lbl_dir = os.path.join(obb_data_folder, 'train', 'labels')
    val_img_dir = os.path.join(obb_data_folder, 'val', 'images')
    val_lbl_dir = os.path.join(obb_data_folder, 'val', 'labels')

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    print("Linking files to train/val directories (using symlinks to save disk space)...")

    def create_link(src, dst):
        """Create a symlink, or copy if symlinks not supported."""
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            # Symlinks may require admin on Windows, fall back to copy
            shutil.copy2(src, dst)

    # Link files (no duplication)
    for img in train_images:
        src = os.path.join(obb_images_dir, img)
        dst = os.path.join(train_img_dir, img)
        lbl = os.path.splitext(img)[0] + '.txt'
        src_lbl = os.path.join(obb_labels_dir, lbl)
        dst_lbl = os.path.join(train_lbl_dir, lbl)
        if os.path.exists(src):
            create_link(src, dst)
        if os.path.exists(src_lbl):
            create_link(src_lbl, dst_lbl)

    for img in val_images:
        src = os.path.join(obb_images_dir, img)
        dst = os.path.join(val_img_dir, img)
        lbl = os.path.splitext(img)[0] + '.txt'
        src_lbl = os.path.join(obb_labels_dir, lbl)
        dst_lbl = os.path.join(val_lbl_dir, lbl)
        if os.path.exists(src):
            create_link(src, dst)
        if os.path.exists(src_lbl):
            create_link(src_lbl, dst_lbl)

    # Create dataset.yaml
    dataset_yaml = os.path.join(obb_data_folder, 'dataset.yaml')
    config = {
        'path': os.path.abspath(obb_data_folder),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'panel'}
    }
    with open(dataset_yaml, 'w') as f:
        yaml.dump(config, f)

    print(f"Dataset config saved to: {dataset_yaml}")
    print()
    print('Starting training...')
    print('=' * 60)

    # Train
    # Options: yolov8n-obb.pt (fastest), yolov8s-obb.pt (balanced), yolov8m-obb.pt (accurate)
    model = YOLO('yolov8s-obb.pt')  # Small model - better accuracy than nano
    model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        patience=15,  # Stop early if no improvement for 15 epochs
        project=os.path.join(obb_data_folder, 'runs'),
        name='manga_panels_obb',
        exist_ok=True
    )

    # Copy best model - search multiple possible locations
    # YOLO sometimes nests the output path in unexpected ways
    possible_paths = [
        # Actual nested location that YOLO creates
        os.path.join('runs', 'obb', obb_data_folder, 'runs', 'manga_panels_obb', 'weights', 'best.pt'),
        # Expected location based on project parameter
        os.path.join(obb_data_folder, 'runs', 'manga_panels_obb', 'weights', 'best.pt'),
        # Standard YOLO runs location
        os.path.join('runs', 'obb', 'manga_panels_obb', 'weights', 'best.pt'),
    ]

    # Also search for any best.pt in runs directory
    import glob
    glob_matches = glob.glob('runs/**/best.pt', recursive=True)
    possible_paths.extend(glob_matches)

    best = None
    for path in possible_paths:
        if os.path.exists(path):
            best = path
            print(f"Found model at: {path}")
            break

    os.makedirs('models', exist_ok=True)
    if best and os.path.exists(best):
        shutil.copy2(best, 'models/manga_panels_obb.pt')
        print('=' * 60)
        print('Training complete!')
        print('Model saved to: models/manga_panels_obb.pt')
        print('=' * 60)
        return True
    else:
        print(f"[WARNING] Best model not found in expected locations")
        print("Searched paths:")
        for p in possible_paths:
            print(f"  - {p}")
        return False


if __name__ == '__main__':
    try:
        success = main()
        if not success:
            print("\n[ERROR] Training failed")
            input("\nPress Enter to exit...")
            sys.exit(1)
    except Exception as e:
        import traceback
        print("\n[ERROR] An exception occurred:")
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)

    print("\n")
    input("Press Enter to exit...")
