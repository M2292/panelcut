"""
Train Segmentation v2 Model - AGGRESSIVE MODE

This is the AGGRESSIVE training variant with 10x higher learning rate (0.01 vs 0.001).
Designed for datasets with 100+ user-corrected images where faster convergence and
more aggressive exploration of the solution space is beneficial.

DIFFERENCES FROM STANDARD TRAINING:
- Higher initial learning rate (0.01 vs 0.001) - YOLO default
- More aggressive weight updates = faster convergence
- Better exploration of solution space
- May see drastic improvements even in later epochs
- Can escape local minima more effectively

WHEN TO USE:
- You have 100+ training images
- Standard training feels too slow/gradual
- You want to explore the solution space more aggressively
- You're willing to trade stability for potential breakthroughs

This script trains a YOLOv8-seg (instance segmentation) model ONLY on user-corrected panel data.
Unlike OBB, segmentation can detect arbitrary quadrilaterals with independent corner positions.

Training data comes exclusively from the download zip feature where users
manually correct panel boxes.

Run via train_seg_v2_aggressive.bat or directly: python train_seg_v2_aggressive.py
"""

import os
import sys
import random
import shutil
import glob
from datetime import datetime

sys.path.insert(0, '.')


# Global variables for tracking training progress
_best_fitness = 0.0
_epochs_without_improvement = 0
_patience = 10
_model_copied = False


def backup_existing_model():
    """
    Create a backup of the existing model before training.
    This protects against regression - if new training makes the model worse,
    you can restore from the backup.

    Backups are named: manga_panels_seg_v2_backup_YYYYMMDD_HHMMSS.pt
    """
    existing_model = 'models/manga_panels_seg_v2.pt'

    if not os.path.exists(existing_model):
        print('[BACKUP] No existing model to backup')
        return None

    # Create backup with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f'models/manga_panels_seg_v2_backup_{timestamp}.pt'

    try:
        shutil.copy2(existing_model, backup_name)
        print(f'[BACKUP] Created backup: {backup_name}')

        # List existing backups
        backups = sorted(glob.glob('models/manga_panels_seg_v2_backup_*.pt'))
        if len(backups) > 1:
            print(f'[BACKUP] You now have {len(backups)} backups:')
            for b in backups[-5:]:  # Show last 5
                print(f'         - {os.path.basename(b)}')
            if len(backups) > 5:
                print(f'         ... and {len(backups) - 5} older backups')

        return backup_name
    except Exception as e:
        print(f'[BACKUP] Failed to create backup: {e}')
        return None


def copy_best_model_now():
    """Copy best model immediately - call this as soon as training finishes."""
    global _model_copied
    if _model_copied:
        return True

    seg_data_folder = os.path.join('training_data', 'seg_v2')

    possible_paths = [
        os.path.join('runs', 'segment', seg_data_folder, 'runs', 'manga_panels_seg_v2', 'weights', 'best.pt'),
        os.path.join(seg_data_folder, 'runs', 'manga_panels_seg_v2', 'weights', 'best.pt'),
        os.path.join('runs', 'segment', 'manga_panels_seg_v2', 'weights', 'best.pt'),
    ]

    glob_matches = glob.glob('runs/**/manga_panels_seg_v2/**/best.pt', recursive=True)
    possible_paths.extend(glob_matches)

    best = None
    for path in possible_paths:
        if os.path.exists(path):
            best = path
            break

    os.makedirs('models', exist_ok=True)
    if best and os.path.exists(best):
        shutil.copy2(best, 'models/manga_panels_seg_v2.pt')
        print()
        print('=' * 60)
        print('MODEL COPIED SUCCESSFULLY!')
        print('Saved to: models/manga_panels_seg_v2.pt')
        print('=' * 60)
        print()
        _model_copied = True
        return True
    return False


def on_fit_epoch_end(trainer):
    """Called at the end of each training epoch - track patience and copy model."""
    global _best_fitness, _epochs_without_improvement, _patience

    current_epoch = trainer.epoch + 1
    total_epochs = trainer.epochs

    # Get current fitness (YOLO uses this to track improvement)
    current_fitness = trainer.fitness if hasattr(trainer, 'fitness') else 0

    # Track improvement
    if current_fitness > _best_fitness:
        _best_fitness = current_fitness
        _epochs_without_improvement = 0
    else:
        _epochs_without_improvement += 1

    # Display patience counter
    remaining_patience = _patience - _epochs_without_improvement
    if _epochs_without_improvement > 0:
        print(f'\n>>> Patience: {remaining_patience}/{_patience} (no improvement for {_epochs_without_improvement} epochs)')
    else:
        print(f'\n>>> New best! Patience reset to {_patience}/{_patience}')

    print(f'>>> Progress: Epoch {current_epoch}/{total_epochs}')
    print()


def on_train_end(trainer):
    """Called when training ends - immediately copy the model before any post-processing."""
    print()
    print('=' * 60)
    print('TRAINING FINISHED - Copying model immediately...')
    print('=' * 60)

    # Copy the model RIGHT NOW before any validation/statistics that might fail
    success = copy_best_model_now()

    if success:
        print('Model secured! Any errors after this point will not affect the saved model.')
    else:
        print('[WARNING] Could not find best.pt to copy yet - will try again after validation.')
    print()

def main():
    try:
        from ultralytics import YOLO
        import yaml
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("\nInstall with: pip install ultralytics pyyaml")
        return False

    TRAINING_DATA_FOLDER = 'training_data'
    seg_data_folder = os.path.join(TRAINING_DATA_FOLDER, 'seg_v2')
    seg_images_dir = os.path.join(seg_data_folder, 'images')
    seg_labels_dir = os.path.join(seg_data_folder, 'labels')

    # Check if data exists
    if not os.path.exists(seg_images_dir):
        print(f"[ERROR] Training images directory not found: {seg_images_dir}")
        print("\nSegmentation v2 model requires user-corrected panel data.")
        print("To collect training data:")
        print("  1. Select 'Seg v2' from the model dropdown in the frontend")
        print("  2. Upload manga pages and manually draw/correct panel corners")
        print("  3. Download the ZIP - this automatically saves training data")
        print(f"\nOnce you have at least 5 images in {seg_images_dir}, run this script again.")
        return False

    # Get image files
    image_files = [f for f in os.listdir(seg_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f'Found {len(image_files)} training images')

    if len(image_files) < 5:
        print(f"[ERROR] Need at least 5 training images, found {len(image_files)}")
        print("\nSegmentation v2 model requires user-corrected panel data.")
        print("To collect more training data:")
        print("  1. Select 'Seg v2' from the model dropdown in the frontend")
        print("  2. Upload manga pages and manually draw/correct panel corners")
        print("  3. Download the ZIP - this automatically saves training data")
        print(f"\nCurrent training data: {seg_images_dir}")
        return False

    # Recommend standard training for small datasets
    if len(image_files) < 100:
        print()
        print('!' * 60)
        print('WARNING: Small dataset detected!')
        print('!' * 60)
        print(f'You have {len(image_files)} images. Aggressive training works best with 100+ images.')
        print('Consider using the standard training script (train_seg_v2_model.bat) instead.')
        print()
        choice = input('Continue with aggressive training anyway? (y/n): ').strip().lower()
        if choice != 'y':
            print('Exiting. Use train_seg_v2_model.bat for standard training.')
            return False
        print()

    # Split 90/10 (more training data, less validation)
    random.shuffle(image_files)
    split_idx = max(1, int(len(image_files) * 0.9))
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:] if split_idx < len(image_files) else [image_files[-1]]

    print(f'Train: {len(train_images)}, Val: {len(val_images)}')

    # Create train/val dirs
    train_img_dir = os.path.join(seg_data_folder, 'train', 'images')
    train_lbl_dir = os.path.join(seg_data_folder, 'train', 'labels')
    val_img_dir = os.path.join(seg_data_folder, 'val', 'images')
    val_lbl_dir = os.path.join(seg_data_folder, 'val', 'labels')

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
        src = os.path.join(seg_images_dir, img)
        dst = os.path.join(train_img_dir, img)
        lbl = os.path.splitext(img)[0] + '.txt'
        src_lbl = os.path.join(seg_labels_dir, lbl)
        dst_lbl = os.path.join(train_lbl_dir, lbl)
        if os.path.exists(src):
            create_link(src, dst)
        if os.path.exists(src_lbl):
            create_link(src_lbl, dst_lbl)

    for img in val_images:
        src = os.path.join(seg_images_dir, img)
        dst = os.path.join(val_img_dir, img)
        lbl = os.path.splitext(img)[0] + '.txt'
        src_lbl = os.path.join(seg_labels_dir, lbl)
        dst_lbl = os.path.join(val_lbl_dir, lbl)
        if os.path.exists(src):
            create_link(src, dst)
        if os.path.exists(src_lbl):
            create_link(src_lbl, dst_lbl)

    # Create dataset.yaml
    dataset_yaml = os.path.join(seg_data_folder, 'dataset.yaml')
    config = {
        'path': os.path.abspath(seg_data_folder),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'panel'}
    }
    with open(dataset_yaml, 'w') as f:
        yaml.dump(config, f)

    print(f"Dataset config saved to: {dataset_yaml}")
    print()

    # Ask user for training mode
    existing_model = 'models/manga_panels_seg_v2.pt'
    base_model = 'yolov8s-seg.pt'
    use_incremental = False

    if os.path.exists(existing_model):
        print('=' * 60)
        print('TRAINING MODE SELECTION')
        print('=' * 60)
        print()
        print(f'Found existing model: {existing_model}')
        print()
        print('Choose training mode:')
        print('  1) Incremental Training - Continue from existing model (recommended)')
        print('     • Faster convergence')
        print('     • Preserves learned manga-specific features')
        print('     • Current model will be backed up')
        print()
        print('  2) Train from Scratch - Start fresh from base YOLOv8 model')
        print('     • Slower training')
        print('     • Ignores existing model completely')
        print('     • Current model will be backed up')
        print()

        while True:
            choice = input('Enter choice (1 or 2): ').strip()
            if choice == '1':
                use_incremental = True
                break
            elif choice == '2':
                use_incremental = False
                break
            else:
                print('Invalid choice. Please enter 1 or 2.')
        print()
    else:
        # No existing model, must start from scratch
        print(f'[INFO] No existing model found, will start from base: {base_model}')
        use_incremental = False

    print('=' * 60)
    print('Starting Segmentation v2 training - AGGRESSIVE MODE')
    print('This model is trained ONLY on user-corrected panel data')
    print('Segmentation can detect arbitrary quadrilaterals (not just rotated rectangles)')
    print()
    print('AGGRESSIVE SETTINGS:')
    print('  • Learning rate: 0.01 (10x higher than standard)')
    print('  • More aggressive weight updates')
    print('  • Better exploration of solution space')
    print(f'  • Patience: {_patience} epochs (will stop early if no improvement)')
    print('=' * 60)
    print()

    # Backup existing model before training (whether incremental or from scratch)
    # This protects against regression in both cases
    if os.path.exists(existing_model):
        backup_existing_model()
        print()

    # Load model based on user choice
    if use_incremental and os.path.exists(existing_model):
        print(f'[INCREMENTAL] Using existing model: {existing_model}')
        print('[INCREMENTAL] Training will build on previous learning')
        model = YOLO(existing_model)
    else:
        print(f'[FRESH START] Starting from base model: {base_model}')
        if os.path.exists(existing_model):
            print('[FRESH START] Existing model will be ignored (but backed up)')
        else:
            print('[FRESH START] This is the first training run')
        model = YOLO(base_model)

    # Register callbacks to copy model immediately when training ends
    # This ensures the model is saved before any post-training statistics that might fail
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
    model.add_callback('on_train_end', on_train_end)

    model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,  # Standard resolution (fast training)
        patience=_patience,  # Stop early if no improvement (default: 10 epochs)
        project=os.path.join(seg_data_folder, 'runs'),
        name='manga_panels_seg_v2',
        exist_ok=True,
        # Tighter training for precise bounding boxes
        box=10.0,  # Increase box loss weight (default 7.5) for tighter boxes
        lr0=0.01,  # AGGRESSIVE: Standard YOLO learning rate (10x higher than conservative)
        lrf=0.01,  # Final learning rate factor
        # Augmentation for better generalization
        degrees=5.0,  # Small rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.3,  # Scale augmentation
        flipud=0.5,  # Vertical flip (manga can be read either way)
        mosaic=0.5,  # Mosaic augmentation (reduced from default 1.0)
        copy_paste=0.1,  # Copy-paste augmentation for segmentation
    )

    # Try to copy model again if callback didn't succeed (fallback)
    if not _model_copied:
        print("\nAttempting fallback model copy...")
        copy_best_model_now()

    if _model_copied:
        print('=' * 60)
        print('AGGRESSIVE TRAINING COMPLETE!')
        print('Model saved to: models/manga_panels_seg_v2.pt')
        print()
        print('This Segmentation v2 model can detect arbitrary quadrilaterals')
        print('with each corner positioned independently (not just rotated rectangles).')
        print()
        print('Trained with aggressive settings (lr0=0.01) for faster convergence.')
        print('=' * 60)
        return True
    else:
        print(f"[WARNING] Best model not found in expected locations")
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
