"""
YOLO Training Script for Manga Panel Detection

This script helps you train a custom YOLO model for manga panel detection.
Supports training on custom datasets or fine-tuning on Manga109.
"""

import os
import argparse
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        from ultralytics import YOLO
        print("[OK] Ultralytics YOLO installed")
        return True
    except ImportError:
        print("[ERROR] Ultralytics not installed. Install with:")
        print("  pip install ultralytics")
        return False


def create_sample_yaml(output_path: str, data_path: str):
    """Create a sample YAML configuration file for training."""
    yaml_content = f"""# Manga Panel Detection Dataset Configuration
# Modify paths to match your dataset location

path: {data_path}
train: images/train
val: images/val
test: images/test  # optional

# Classes for manga panel detection
names:
  0: panel      # Comic panel/frame
  1: text       # Text region
  2: balloon    # Speech balloon
  3: face       # Character face
  4: body       # Character body

# Notes:
# - For basic panel detection, you only need class 0 (panel)
# - Add more classes if you want to detect text, faces, etc.
# - Image annotations should be in YOLO format (txt files)
"""
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created sample config: {output_path}")


def train_model(
    data_yaml: str,
    model_name: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/manga",
    name: str = "panel_detector"
):
    """
    Train a YOLO model for manga panel detection.

    Args:
        data_yaml: Path to dataset YAML configuration
        model_name: Base model to fine-tune (yolov8n.pt, yolov8s.pt, etc.)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        project: Project directory for outputs
        name: Experiment name
    """
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print("Manga Panel Detection - YOLO Training")
    print(f"{'='*60}")
    print(f"Base model: {model_name}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"{'='*60}\n")

    # Load model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        patience=20,  # Early stopping patience
        save=True,
        plots=True,
        verbose=True
    )

    # Get best model path
    best_model = Path(project) / name / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best model saved to: {best_model}")
    print(f"{'='*60}")

    # Copy to models directory
    models_dir = Path(__file__).parent / "models"
    if models_dir.exists():
        import shutil
        dest = models_dir / "manga_panels.pt"
        shutil.copy(best_model, dest)
        print(f"Copied to: {dest}")

    return str(best_model)


def validate_model(model_path: str, data_yaml: str):
    """Validate a trained model on the test set."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.val(data=data_yaml)

    print("\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")

    return results


def export_model(model_path: str, format: str = "onnx"):
    """Export model to different formats for deployment."""
    from ultralytics import YOLO

    model = YOLO(model_path)

    # Export
    export_path = model.export(format=format)
    print(f"Exported model to: {export_path}")

    return export_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for manga panel detection")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data", required=True, help="Path to data YAML")
    train_parser.add_argument("--model", default="yolov8n.pt", help="Base model")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--name", default="panel_detector")

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate a model")
    val_parser.add_argument("--model", required=True, help="Path to model")
    val_parser.add_argument("--data", required=True, help="Path to data YAML")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--model", required=True, help="Path to model")
    export_parser.add_argument("--format", default="onnx", help="Export format")

    # Create config command
    config_parser = subparsers.add_parser("create-config", help="Create sample YAML")
    config_parser.add_argument("--output", default="manga_panels.yaml")
    config_parser.add_argument("--data-path", default="./dataset")

    args = parser.parse_args()

    if not check_dependencies():
        return

    if args.command == "train":
        train_model(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name
        )
    elif args.command == "validate":
        validate_model(args.model, args.data)
    elif args.command == "export":
        export_model(args.model, args.format)
    elif args.command == "create-config":
        create_sample_yaml(args.output, args.data_path)
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("Quick Start:")
        print("="*60)
        print("1. Create dataset config:")
        print("   python train_yolo.py create-config --output my_data.yaml")
        print("\n2. Train model:")
        print("   python train_yolo.py train --data my_data.yaml --epochs 50")
        print("\n3. Validate:")
        print("   python train_yolo.py validate --model models/manga_panels.pt --data my_data.yaml")


if __name__ == "__main__":
    main()
