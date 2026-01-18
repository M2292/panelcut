"""
Test OBB Detection - Visual Output

This script runs the trained OBB model on sample images and saves
visualization results so you can verify the detections are correct.
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
        from yolo_detector import OBBDetector
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("\nInstall with: pip install ultralytics opencv-python numpy")
        return False

    # Configuration
    OBB_MODEL = 'models/manga_panels_obb.pt'
    INPUT_DIR = 'bootstrap_images'
    OUTPUT_DIR = 'test_obb_results'
    NUM_SAMPLES = 5

    # Check model exists
    if not os.path.exists(OBB_MODEL):
        print(f"[ERROR] OBB model not found: {OBB_MODEL}")
        print("Run train_obb_model.bat first to train the model.")
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
    print("OBB Detection Test")
    print("=" * 60)
    print(f"Model: {OBB_MODEL}")
    print(f"Input directory: {input_path.absolute()}")
    print(f"Total images available: {len(image_files)}")
    print(f"Testing on: {NUM_SAMPLES} random samples")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print("\nLoading OBB model...")
    detector = OBBDetector(model_path=OBB_MODEL, confidence=0.5)
    print("Model loaded!")

    # Select random samples
    if len(image_files) > NUM_SAMPLES:
        samples = random.sample(image_files, NUM_SAMPLES)
    else:
        samples = image_files[:NUM_SAMPLES]

    print(f"\nProcessing {len(samples)} images...")
    print("-" * 60)

    results_summary = []

    for i, img_path in enumerate(samples):
        try:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  [{i+1}] SKIP (unreadable): {img_path.name}")
                continue

            # Detect panels
            panels = detector.detect_panels_only(image)

            # Draw detections
            viz = detector.draw_detections(image, panels, show_confidence=True, show_angle=True)

            # Add info text at top
            info_text = f"File: {img_path.name} | Panels: {len(panels)}"
            cv2.rectangle(viz, (0, 0), (viz.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(viz, info_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Save result
            output_name = f"test_{i+1:02d}_{img_path.stem}.png"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            cv2.imwrite(output_path, viz)

            # Collect panel info
            panel_info = []
            for j, p in enumerate(panels):
                angle_str = f" (rotated {p.angle:.0f} deg)" if abs(p.angle) > 2 else ""
                panel_info.append(f"    Panel {j+1}: conf={p.confidence:.2f}{angle_str}")

            results_summary.append({
                'file': img_path.name,
                'panels': len(panels),
                'output': output_name,
                'details': panel_info
            })

            print(f"  [{i+1}/{len(samples)}] {img_path.name} -> {len(panels)} panels -> {output_name}")

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
    print("Open the 'test_obb_results' folder to view the detection images.")
    print("Each image shows the detected panels with colored boxes.")
    print("Rotated panels will show their angle (e.g., '45 deg').")
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
