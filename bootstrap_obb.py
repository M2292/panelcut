"""
Bootstrap OBB Training Data from Existing YOLO Model

This script processes manga images using your existing YOLO model and saves
the detections as OBB (Oriented Bounding Box) format training data.

Usage:
    1. Place manga page images in the 'bootstrap_images' folder
    2. Run: python bootstrap_obb.py
    3. Or double-click: bootstrap_obb.bat

The script will:
    - Run your existing YOLO model on all images
    - Save detections as OBB format (axis-aligned, angle=0)
    - Create training data ready for OBB fine-tuning

After bootstrapping, your diagonal corrections during normal app usage
will add rotation knowledge on top of this base.

Recommended image counts:
    - Minimum: 20 images (basic training)
    - Good: 50 images (decent generalization)
    - Better: 100+ images (robust model)
    - Best: 200+ images with variety (production quality)
"""

import os
import sys
import argparse
from pathlib import Path
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        return True
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install ultralytics opencv-python numpy")
        return False


def find_yolo_model():
    """Find the best available YOLO model."""
    models_dir = Path(__file__).parent / 'models'

    # Priority order for models
    model_priority = [
        'manga109_yolo.pt',
        'manga_panels.pt',
        'manga_panels_finetuned.pt',
        'yolov8n.pt',
        'yolov8s.pt'
    ]

    for model_name in model_priority:
        model_path = models_dir / model_name
        if model_path.exists():
            return str(model_path)

    # Fall back to downloading yolov8n
    return 'yolov8n.pt'


def get_image_files(input_dir: Path, recursive: bool = True) -> list:
    """
    Get all image files from directory (and subdirectories if recursive).

    Args:
        input_dir: Root directory to search
        recursive: If True, search all subdirectories too

    Returns:
        List of image file paths
    """
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    files = []

    # Use ** for recursive glob, * for single directory
    pattern = '**/*' if recursive else '*'

    for ext in extensions:
        files.extend(input_dir.glob(f'{pattern}{ext}'))
        files.extend(input_dir.glob(f'{pattern}{ext.upper()}'))

    return sorted(files)


def bootstrap_obb_data(
    input_dir: str = 'bootstrap_images',
    output_dir: str = 'training_data/obb',
    model_path: str = None,
    confidence: float = 0.5,
    max_images: int = None,
    skip_existing: bool = True,
    recursive: bool = True
):
    """
    Bootstrap OBB training data from images using existing YOLO model.

    Args:
        input_dir: Directory containing manga page images (scans subfolders too)
        output_dir: Output directory for OBB training data
        model_path: Path to YOLO model (auto-detect if None)
        confidence: Detection confidence threshold
        max_images: Maximum images to process (None = all)
        skip_existing: Skip images that already have OBB labels
        recursive: If True, scan all subfolders within input_dir
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Find model
    if model_path is None:
        model_path = find_yolo_model()

    if not os.path.exists(model_path) and not model_path.startswith('yolov'):
        print(f"[ERROR] Model not found: {model_path}")
        return False

    print("=" * 60)
    print("OBB Training Data Bootstrap")
    print("=" * 60)
    print(f"Input directory: {input_path.absolute()}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {confidence}")
    print(f"Recursive scan: {recursive}")
    print("=" * 60)

    # Get image files (recursively if enabled)
    image_files = get_image_files(input_path, recursive=recursive)

    if not image_files:
        print(f"\n[ERROR] No images found in '{input_dir}'")
        if recursive:
            print("(Searched all subfolders)")
        print(f"\nPlease place manga page images in: {input_path.absolute()}")
        print("You can organize them in subfolders - they will all be found.")
        print("\nSupported formats: .jpg, .jpeg, .png, .webp, .bmp, .tiff")
        return False

    # Count images per subfolder for info
    if recursive:
        folder_counts = {}
        for img in image_files:
            rel_folder = img.parent.relative_to(input_path) if img.parent != input_path else Path('.')
            folder_counts[str(rel_folder)] = folder_counts.get(str(rel_folder), 0) + 1

        if len(folder_counts) > 1:
            print(f"\nImages by folder:")
            for folder, count in sorted(folder_counts.items()):
                display_name = folder if folder != '.' else '(root)'
                print(f"  {display_name}: {count} images")

    if max_images:
        image_files = image_files[:max_images]

    print(f"\nFound {len(image_files)} images to process")

    # Load model
    print(f"\nLoading YOLO model...")
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")

    # Detect which class is "frame" or "panel"
    panel_class_id = None
    if hasattr(model, 'names'):
        print(f"Model classes: {model.names}")
        for cls_id, cls_name in model.names.items():
            if cls_name.lower() in ['frame', 'panel', 'panels']:
                panel_class_id = cls_id
                print(f"Using class '{cls_name}' (id={cls_id}) for panel detection")
                break

    if panel_class_id is None:
        print("WARNING: Could not find 'frame' or 'panel' class, using class 0")
        panel_class_id = 0

    # Process images
    processed = 0
    skipped = 0
    total_panels = 0
    errors = 0

    print(f"\nProcessing images...")
    print("-" * 60)

    for i, img_path in enumerate(image_files):
        try:
            # Generate output filenames with folder prefix to avoid collisions
            # e.g., "MangaA/chapter1/page01.jpg" -> "MangaA_chapter1_page01.png"
            try:
                rel_path = img_path.relative_to(input_path)
                # Create unique name from path: folder1_folder2_filename
                parts = list(rel_path.parent.parts) + [img_path.stem]
                unique_stem = '_'.join(parts) if len(parts) > 1 else img_path.stem
            except ValueError:
                unique_stem = img_path.stem

            # Clean up any problematic characters
            unique_stem = unique_stem.replace(' ', '_').replace('-', '_')

            out_image = images_dir / f"{unique_stem}.png"
            out_label = labels_dir / f"{unique_stem}.txt"

            # Skip if already processed
            if skip_existing and out_label.exists():
                skipped += 1
                continue

            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  [{i+1}/{len(image_files)}] SKIP (unreadable): {img_path.name}")
                errors += 1
                continue

            img_height, img_width = image.shape[:2]

            # Run detection - filter to panel class only
            results = model(image, conf=confidence, classes=[panel_class_id], verbose=False)

            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                print(f"  [{i+1}/{len(image_files)}] SKIP (no panels): {img_path.name}")
                continue

            boxes = results[0].boxes
            num_panels = len(boxes)

            # Convert to OBB format (axis-aligned boxes as 4 corners)
            obb_labels = []
            for box in boxes:
                # Get bbox coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Create 4 corners (clockwise from top-left)
                corners = [
                    (x1, y1),  # top-left
                    (x2, y1),  # top-right
                    (x2, y2),  # bottom-right
                    (x1, y2)   # bottom-left
                ]

                # Normalize coordinates
                obb_parts = ['0']  # class_id = 0 (panel)
                for cx, cy in corners:
                    nx = max(0, min(1, cx / img_width))
                    ny = max(0, min(1, cy / img_height))
                    obb_parts.append(f"{nx:.6f}")
                    obb_parts.append(f"{ny:.6f}")

                obb_labels.append(' '.join(obb_parts))

            # Save image (copy or convert to PNG)
            if img_path.suffix.lower() == '.png':
                shutil.copy2(img_path, out_image)
            else:
                cv2.imwrite(str(out_image), image)

            # Save OBB labels
            with open(out_label, 'w') as f:
                f.write('\n'.join(obb_labels))

            processed += 1
            total_panels += num_panels

            # Show relative path for clarity
            try:
                display_path = img_path.relative_to(input_path)
            except ValueError:
                display_path = img_path.name
            print(f"  [{i+1}/{len(image_files)}] OK: {display_path} -> {num_panels} panels")

        except Exception as e:
            try:
                display_path = img_path.relative_to(input_path)
            except ValueError:
                display_path = img_path.name
            print(f"  [{i+1}/{len(image_files)}] ERROR: {display_path} - {e}")
            errors += 1

    # Summary
    print("-" * 60)
    print(f"\nBootstrap Complete!")
    print("=" * 60)
    print(f"Images processed: {processed}")
    print(f"Images skipped (existing): {skipped}")
    print(f"Images with errors: {errors}")
    print(f"Total panels detected: {total_panels}")
    print(f"\nOutput location: {output_path.absolute()}")
    print("=" * 60)

    # Training recommendations
    print("\n📊 Training Data Assessment:")
    if processed < 10:
        print("   ⚠️  Very few samples. Add more images for better results.")
        print("   Recommendation: At least 20 images minimum.")
    elif processed < 20:
        print("   ⚠️  Minimal training data. Model may overfit.")
        print("   Recommendation: Add more diverse manga pages.")
    elif processed < 50:
        print("   ✓  Decent amount of data for initial training.")
        print("   Tip: Add diagonal corrections during app usage to improve.")
    elif processed < 100:
        print("   ✓  Good training data. Should produce reasonable results.")
    else:
        print("   ✓  Excellent! Sufficient data for robust training.")

    print("\n🚀 Next Steps:")
    print("   1. Start the app: python app.py")
    print("   2. Go to 'Training' section in the UI")
    print("   3. Click 'Train OBB Model' button")
    print("   4. Use app normally - diagonal corrections will improve the model")

    return True


def create_input_folder():
    """Create the bootstrap_images folder if it doesn't exist."""
    input_dir = Path(__file__).parent / 'bootstrap_images'
    input_dir.mkdir(exist_ok=True)

    # Create a README
    readme_path = input_dir / 'README.txt'
    if not readme_path.exists():
        readme_path.write_text("""
==============================================
BOOTSTRAP IMAGES FOLDER
==============================================

Place your manga page images in this folder.
You can organize them in SUBFOLDERS - they will all be found!

Example structure:
  bootstrap_images/
    MangaA/
      page001.jpg
      page002.jpg
    MangaB/
      chapter1/
        page01.png
        page02.png

Supported formats:
  - .jpg, .jpeg
  - .png
  - .webp
  - .bmp
  - .tiff

Recommendations:
  - Use high-quality scans/rips
  - Include variety (different manga, layouts)
  - 20 images minimum, 50-100 recommended
  - Mix of simple and complex page layouts

After adding images, run:
  - Double-click: bootstrap_obb.bat
  - Or command: python bootstrap_obb.py

==============================================
""")

    return input_dir


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap OBB training data from existing YOLO model"
    )
    parser.add_argument(
        '--input', '-i',
        default='bootstrap_images',
        help='Input directory with manga images (default: bootstrap_images)'
    )
    parser.add_argument(
        '--output', '-o',
        default='training_data/obb',
        help='Output directory for OBB data (default: training_data/obb)'
    )
    parser.add_argument(
        '--model', '-m',
        default=None,
        help='YOLO model path (default: auto-detect best available)'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--max-images', '-n',
        type=int,
        default=None,
        help='Maximum images to process (default: all)'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Overwrite existing OBB labels'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Only scan top-level folder, not subfolders'
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        input("\nPress Enter to exit...")
        sys.exit(1)

    # Create input folder if needed
    create_input_folder()

    # Run bootstrap
    success = bootstrap_obb_data(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        confidence=args.confidence,
        max_images=args.max_images,
        skip_existing=not args.force,
        recursive=not args.no_recursive
    )

    if not success:
        input("\nPress Enter to exit...")
        sys.exit(1)

    print("\n")
    input("Press Enter to exit...")


if __name__ == '__main__':
    main()
