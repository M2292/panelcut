"""
Test YOLO Bootstrap Detections - Visual Output

This script runs the STANDARD YOLO model (the same one used by bootstrap_obb.bat)
on sample images and saves visualization results so you can verify that the
detections being used for OBB training data are correct.
"""

import os
import sys
import random
from pathlib import Path

sys.path.insert(0, '.')


def main():
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("\nInstall with: pip install ultralytics opencv-python numpy")
        return False

    # Configuration
    INPUT_DIR = 'bootstrap_images'
    OUTPUT_DIR = 'test_yolo_bootstrap_results'
    NUM_SAMPLES = 5
    CONFIDENCE = 0.5

    # Find the YOLO model (same logic as bootstrap_obb.py)
    models_dir = Path('models')
    model_priority = [
        'manga109_yolo.pt',
        'manga_panels.pt',
        'manga_panels_finetuned.pt',
        'yolov8n.pt',
    ]

    yolo_model = None
    for model_name in model_priority:
        model_path = models_dir / model_name
        if model_path.exists():
            yolo_model = str(model_path)
            break

    if yolo_model is None:
        print("[ERROR] No YOLO model found in 'models' folder")
        print("Expected one of:", model_priority)
        return False

    # Get image files from bootstrap_images
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        return False

    # Find all images recursively
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f'**/*{ext}'))
        image_files.extend(input_path.glob(f'**/*{ext.upper()}'))

    if not image_files:
        print(f"[ERROR] No images found in {INPUT_DIR}")
        return False

    print("=" * 60)
    print("YOLO Bootstrap Detection Test")
    print("=" * 60)
    print(f"Model: {yolo_model}")
    print(f"  (This is the model used by bootstrap_obb.bat)")
    print(f"Input directory: {input_path.absolute()}")
    print(f"Total images available: {len(image_files)}")
    print(f"Testing on: {NUM_SAMPLES} random samples")
    print(f"Confidence threshold: {CONFIDENCE}")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print("\nLoading YOLO model...")
    model = YOLO(yolo_model)
    print(f"Model loaded: {yolo_model}")

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

    # Select random samples
    if len(image_files) > NUM_SAMPLES:
        samples = random.sample(image_files, NUM_SAMPLES)
    else:
        samples = image_files[:NUM_SAMPLES]

    print(f"\nProcessing {len(samples)} images...")
    print("-" * 60)

    results_summary = []
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (128, 0, 255), (255, 128, 0)
    ]

    for i, img_path in enumerate(samples):
        try:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  [{i+1}] SKIP (unreadable): {img_path.name}")
                continue

            # Run detection - filter to panel class only (same as bootstrap_obb.py)
            results = model(image, conf=CONFIDENCE, classes=[panel_class_id], verbose=False)

            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                print(f"  [{i+1}/{len(samples)}] {img_path.name} -> 0 panels detected")
                # Still save the image with "no panels" note
                viz = image.copy()
                cv2.rectangle(viz, (0, 0), (viz.shape[1], 50), (0, 0, 0), -1)
                cv2.putText(viz, f"File: {img_path.name} | NO PANELS DETECTED",
                           (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                output_name = f"test_{i+1:02d}_{img_path.stem}.png"
                output_path = os.path.join(OUTPUT_DIR, output_name)
                cv2.imwrite(output_path, viz)
                results_summary.append({
                    'file': img_path.name,
                    'panels': 0,
                    'output': output_name,
                    'details': []
                })
                continue

            boxes = results[0].boxes
            num_panels = len(boxes)

            # Draw detections
            viz = image.copy()
            panel_info = []

            for j, box in enumerate(boxes):
                color = colors[j % len(colors)]

                # Get bbox coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())

                # Draw rectangle
                cv2.rectangle(viz, (x1, y1), (x2, y2), color, 3)

                # Draw label
                label = f"{j+1} {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(viz, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
                cv2.putText(viz, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                panel_info.append(f"    Panel {j+1}: conf={conf:.2f}, bbox=({x1},{y1})-({x2},{y2})")

            # Add info text at top
            info_text = f"File: {img_path.name} | Panels: {num_panels}"
            cv2.rectangle(viz, (0, 0), (viz.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(viz, info_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Save result
            output_name = f"test_{i+1:02d}_{img_path.stem}.png"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            cv2.imwrite(output_path, viz)

            results_summary.append({
                'file': img_path.name,
                'panels': num_panels,
                'output': output_name,
                'details': panel_info
            })

            print(f"  [{i+1}/{len(samples)}] {img_path.name} -> {num_panels} panels -> {output_name}")

        except Exception as e:
            print(f"  [{i+1}] ERROR: {img_path.name} - {e}")

    # Summary
    print("-" * 60)
    print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

    print("\nDetailed Results:")
    print("-" * 60)
    for r in results_summary:
        print(f"\n{r['file']}:")
        print(f"  Panels detected: {r['panels']}")
        print(f"  Output: {r['output']}")
        if r['details']:
            for detail in r['details']:
                print(detail)

    print("\n" + "=" * 60)
    print("WHAT TO CHECK:")
    print("-" * 60)
    print("1. Are all panels correctly detected?")
    print("2. Are there any FALSE POSITIVES (boxes that aren't panels)?")
    print("3. Are there any MISSED PANELS (panels without boxes)?")
    print("4. Are the bounding boxes TIGHT around the panels?")
    print("")
    print("If detections look wrong, the OBB training data will be wrong too!")
    print("=" * 60)

    return True


if __name__ == '__main__':
    try:
        success = main()
        if not success:
            print("\n[ERROR] Test failed")
    except Exception as e:
        import traceback
        print("\n[ERROR] An exception occurred:")
        traceback.print_exc()

    print("\n")
    input("Press Enter to exit...")
