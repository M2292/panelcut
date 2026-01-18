"""
YOLO-based Manga Panel Detection

Uses Ultralytics YOLOv8/v11 for ML-based panel detection.
Supports pre-trained models or custom-trained models on Manga109 dataset.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Optional YOLO import - gracefully handle if not installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None


@dataclass
class YOLOPanel:
    """Represents a panel detected by YOLO"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    contour: np.ndarray  # For compatibility with existing Panel class


@dataclass
class OBBPanel:
    """Represents a panel detected by YOLO-OBB (Oriented Bounding Box)"""
    corners: np.ndarray  # 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    confidence: float
    class_id: int
    class_name: str
    angle: float  # Rotation angle in degrees
    center: Tuple[float, float]  # Center point (cx, cy)
    size: Tuple[float, float]  # Width and height of the rotated box

    @property
    def contour(self) -> np.ndarray:
        """Return corners as OpenCV contour format for compatibility"""
        return self.corners.reshape((-1, 1, 2)).astype(np.int32)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return axis-aligned bounding box (x1, y1, x2, y2) for compatibility"""
        xs = self.corners[:, 0]
        ys = self.corners[:, 1]
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


class YOLODetector:
    """
    YOLO-based manga panel detector.

    Supports:
    - Pre-trained models (manga109_yolo, custom trained)
    - Multiple YOLO versions (v8, v11)
    - Batch processing
    - Confidence thresholding
    """

    # Known model paths (can be URLs or local paths)
    PRETRAINED_MODELS = {
        'manga109_yolo': 'models/manga109_yolo.pt',  # Best manga model - detects frame/body/face/text
        'manga_panels': 'models/manga_panels.pt',
        'yolov8n': 'yolov8n.pt',  # Base model for fine-tuning
        'yolov8s': 'yolov8s.pt',
        'yolov8m': 'yolov8m.pt',
    }

    # Default model to use
    DEFAULT_MODEL = 'models/manga109_yolo.pt'

    # Class names for manga detection (manga109_yolo model)
    MANGA_CLASSES = {
        0: 'body',
        1: 'face',
        2: 'frame',  # This is the panel class we want
        3: 'text',
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'auto'
    ):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model (.pt file) or model name
            confidence: Minimum confidence threshold (0-1)
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('auto', 'cpu', 'cuda', '0', '1', etc.)
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "Ultralytics YOLO not installed. "
                "Install with: pip install ultralytics"
            )

        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.model_path = model_path

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Load a YOLO model.

        Args:
            model_path: Path to model file or model name from PRETRAINED_MODELS

        Returns:
            True if loaded successfully
        """
        # Check if it's a known model name
        if model_path in self.PRETRAINED_MODELS:
            model_path = self.PRETRAINED_MODELS[model_path]

        # Check if local file exists
        if not os.path.exists(model_path):
            # Try in models directory
            models_dir = Path(__file__).parent / 'models'
            alt_path = models_dir / model_path
            if alt_path.exists():
                model_path = str(alt_path)
            elif not model_path.startswith('yolov'):
                # Not a standard model name and file doesn't exist
                print(f"[YOLO] Model not found: {model_path}")
                return False

        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            print(f"[YOLO] Loaded model: {model_path}")
            return True
        except Exception as e:
            print(f"[YOLO] Error loading model: {e}")
            return False

    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None,
        verbose: bool = False
    ) -> List[YOLOPanel]:
        """
        Detect panels in a manga page.

        Args:
            image: BGR image (OpenCV format)
            classes: List of class IDs to detect (None = all)
            verbose: Print detection info

        Returns:
            List of YOLOPanel objects
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Run inference
        results = self.model(
            image,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=classes,
            verbose=verbose,
            device=self.device if self.device != 'auto' else None
        )

        panels = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # Get bounding box
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Get class name
                cls_name = self.MANGA_CLASSES.get(cls_id, f'class_{cls_id}')
                if hasattr(self.model, 'names') and cls_id in self.model.names:
                    cls_name = self.model.names[cls_id]

                # Create contour for compatibility with existing system
                contour = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], dtype=np.int32).reshape((-1, 1, 2))

                panels.append(YOLOPanel(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    contour=contour
                ))

        return panels

    def detect_panels_only(
        self,
        image: np.ndarray,
        panel_class_ids: List[int] = [2]  # Class 2 = 'frame' in manga109_yolo
    ) -> List[YOLOPanel]:
        """
        Detect only panel/frame objects.

        Args:
            image: BGR image
            panel_class_ids: Class IDs that represent panels (default: [2] for 'frame')

        Returns:
            List of panel detections
        """
        all_detections = self.detect(image)
        return [p for p in all_detections if p.class_id in panel_class_ids]

    def create_panel_mask(
        self,
        image: np.ndarray,
        panels: List[YOLOPanel]
    ) -> np.ndarray:
        """
        Create a binary mask from YOLO panel detections.

        Args:
            image: Original image (for dimensions)
            panels: List of detected panels

        Returns:
            Binary mask (255 = panel, 0 = gutter)
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for panel in panels:
            cv2.fillPoly(mask, [panel.contour], 255)

        return mask

    def draw_detections(
        self,
        image: np.ndarray,
        panels: List[YOLOPanel],
        show_confidence: bool = True,
        show_class: bool = True
    ) -> np.ndarray:
        """
        Draw detection boxes on image.

        Args:
            image: BGR image
            panels: List of detected panels
            show_confidence: Show confidence scores
            show_class: Show class names

        Returns:
            Image with drawn detections
        """
        viz = image.copy()
        colors = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0),
            (0, 255, 255), (255, 0, 255), (255, 255, 0),
            (128, 0, 255), (255, 128, 0)
        ]

        for i, panel in enumerate(panels):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = panel.bbox

            # Draw box
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 3)

            # Build label
            label_parts = [str(i + 1)]
            if show_class:
                label_parts.append(panel.class_name)
            if show_confidence:
                label_parts.append(f'{panel.confidence:.2f}')
            label = ' '.join(label_parts)

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                viz,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 10, y1),
                color, -1
            )

            # Draw label text
            cv2.putText(
                viz, label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )

        return viz


class OBBDetector:
    """
    YOLO-OBB based manga panel detector for rotated/diagonal boxes.

    Uses YOLOv8-OBB models that can detect oriented bounding boxes,
    which is essential for diagonal manga panels.
    """

    DEFAULT_MODEL = 'models/manga_panels_obb.pt'

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'auto'
    ):
        """
        Initialize OBB detector.

        Args:
            model_path: Path to OBB YOLO model (.pt file)
            confidence: Minimum confidence threshold (0-1)
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('auto', 'cpu', 'cuda', etc.)
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "Ultralytics YOLO not installed. "
                "Install with: pip install ultralytics"
            )

        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.model_path = model_path

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Load a YOLO-OBB model.

        Args:
            model_path: Path to OBB model file

        Returns:
            True if loaded successfully
        """
        if not os.path.exists(model_path):
            models_dir = Path(__file__).parent / 'models'
            alt_path = models_dir / model_path
            if alt_path.exists():
                model_path = str(alt_path)
            elif not model_path.startswith('yolov'):
                print(f"[OBB] Model not found: {model_path}")
                return False

        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            print(f"[OBB] Loaded model: {model_path}")
            return True
        except Exception as e:
            print(f"[OBB] Error loading model: {e}")
            return False

    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None,
        verbose: bool = False
    ) -> List[OBBPanel]:
        """
        Detect oriented bounding boxes in a manga page.

        Args:
            image: BGR image (OpenCV format)
            classes: List of class IDs to detect (None = all)
            verbose: Print detection info

        Returns:
            List of OBBPanel objects with rotated boxes
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Run inference
        results = self.model(
            image,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=classes,
            verbose=verbose,
            device=self.device if self.device != 'auto' else None
        )

        panels = []

        if len(results) > 0 and results[0].obb is not None:
            obb_results = results[0].obb

            for i in range(len(obb_results)):
                # Get OBB data
                # OBB format: xyxyxyxy (4 corner points) or xywhr (center, size, rotation)
                if hasattr(obb_results, 'xyxyxyxy'):
                    # 4 corner points format
                    corners = obb_results.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                elif hasattr(obb_results, 'xywhr'):
                    # Center, width, height, rotation format
                    xywhr = obb_results.xywhr[i].cpu().numpy()
                    cx, cy, w, h, r = xywhr
                    corners = self._xywhr_to_corners(cx, cy, w, h, r)
                else:
                    continue

                conf = float(obb_results.conf[i].cpu().numpy())
                cls_id = int(obb_results.cls[i].cpu().numpy())

                # Get class name
                cls_name = 'panel'
                if hasattr(self.model, 'names') and cls_id in self.model.names:
                    cls_name = self.model.names[cls_id]

                # Calculate center, size, and angle from corners
                center = corners.mean(axis=0)
                # Approximate angle from first edge
                edge = corners[1] - corners[0]
                angle = np.degrees(np.arctan2(edge[1], edge[0]))
                # Approximate size
                w = np.linalg.norm(corners[1] - corners[0])
                h = np.linalg.norm(corners[2] - corners[1])

                panels.append(OBBPanel(
                    corners=corners,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    angle=angle,
                    center=(float(center[0]), float(center[1])),
                    size=(float(w), float(h))
                ))

        return panels

    def _xywhr_to_corners(
        self,
        cx: float, cy: float,
        w: float, h: float,
        r: float
    ) -> np.ndarray:
        """Convert center-width-height-rotation to 4 corner points."""
        cos_r = np.cos(r)
        sin_r = np.sin(r)

        # Half dimensions
        hw, hh = w / 2, h / 2

        # Corner offsets before rotation
        offsets = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh]
        ])

        # Rotation matrix
        rot = np.array([
            [cos_r, -sin_r],
            [sin_r, cos_r]
        ])

        # Rotate and translate
        corners = offsets @ rot.T + np.array([cx, cy])
        return corners

    def detect_panels_only(
        self,
        image: np.ndarray,
        panel_class_ids: List[int] = [0]
    ) -> List[OBBPanel]:
        """
        Detect only panel objects.

        Args:
            image: BGR image
            panel_class_ids: Class IDs that represent panels

        Returns:
            List of OBBPanel detections
        """
        all_detections = self.detect(image)
        return [p for p in all_detections if p.class_id in panel_class_ids]

    def create_panel_mask(
        self,
        image: np.ndarray,
        panels: List[OBBPanel]
    ) -> np.ndarray:
        """
        Create a binary mask from OBB panel detections.

        Args:
            image: Original image (for dimensions)
            panels: List of detected OBB panels

        Returns:
            Binary mask (255 = panel, 0 = gutter)
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for panel in panels:
            cv2.fillPoly(mask, [panel.contour], 255)

        return mask

    def draw_detections(
        self,
        image: np.ndarray,
        panels: List[OBBPanel],
        show_confidence: bool = True,
        show_angle: bool = True
    ) -> np.ndarray:
        """
        Draw OBB detection boxes on image.

        Args:
            image: BGR image
            panels: List of detected OBB panels
            show_confidence: Show confidence scores
            show_angle: Show rotation angle

        Returns:
            Image with drawn detections
        """
        viz = image.copy()
        colors = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0),
            (0, 255, 255), (255, 0, 255), (255, 255, 0),
            (128, 0, 255), (255, 128, 0)
        ]

        for i, panel in enumerate(panels):
            color = colors[i % len(colors)]

            # Draw rotated box
            corners_int = panel.corners.astype(np.int32)
            cv2.polylines(viz, [corners_int], True, color, 3)

            # Build label
            label_parts = [str(i + 1)]
            if show_confidence:
                label_parts.append(f'{panel.confidence:.2f}')
            if show_angle and abs(panel.angle) > 2:  # Only show if notably rotated
                label_parts.append(f'{panel.angle:.0f}°')
            label = ' '.join(label_parts)

            # Draw label at center
            cx, cy = int(panel.center[0]), int(panel.center[1])
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                viz,
                (cx - label_w // 2 - 5, cy - label_h // 2 - 5),
                (cx + label_w // 2 + 5, cy + label_h // 2 + 5),
                color, -1
            )
            cv2.putText(
                viz, label,
                (cx - label_w // 2, cy + label_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )

        return viz


def get_available_models() -> Dict[str, str]:
    """
    Get list of available YOLO models (both standard and OBB).

    Returns:
        Dict of model_name -> path/status
    """
    models_dir = Path(__file__).parent / 'models'
    available = {}

    # Check for local models
    if models_dir.exists():
        for pt_file in models_dir.glob('*.pt'):
            # Mark OBB models
            name = pt_file.stem
            if 'obb' in name.lower():
                available[name] = f'{pt_file} (OBB)'
            else:
                available[name] = str(pt_file)

    # Add standard Ultralytics models
    available['yolov8n'] = 'yolov8n.pt (download on first use)'
    available['yolov8s'] = 'yolov8s.pt (download on first use)'
    available['yolov8m'] = 'yolov8m.pt (download on first use)'

    # Add OBB models
    available['yolov8n-obb'] = 'yolov8n-obb.pt (OBB, download on first use)'
    available['yolov8s-obb'] = 'yolov8s-obb.pt (OBB, download on first use)'

    return available


def download_manga_model(model_name: str = 'manga_panels') -> Optional[str]:
    """
    Download a pre-trained manga panel detection model.

    Note: This is a placeholder. In practice, you would:
    1. Train your own model on Manga109 dataset
    2. Use a community-shared model
    3. Fine-tune a base YOLO model

    Args:
        model_name: Name of model to download

    Returns:
        Path to downloaded model or None
    """
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)

    # Placeholder - would need actual model hosting
    print(f"[YOLO] To use manga-specific models, you need to:")
    print("  1. Train a model on Manga109 dataset")
    print("  2. Or fine-tune YOLOv8 on your manga samples")
    print("  3. Place the .pt file in the 'models' directory")
    print()
    print("Training instructions:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('yolov8n.pt')")
    print("  model.train(data='manga_panels.yaml', epochs=100)")

    return None


def is_yolo_available() -> bool:
    """Check if YOLO is available."""
    return YOLO_AVAILABLE


# Convenience function for integration with existing panel_slicer
def detect_panels_yolo(
    image: np.ndarray,
    model_path: str = 'yolov8n.pt',
    confidence: float = 0.5,
    panel_classes: List[int] = [0, 1]
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Detect panels using YOLO (convenience function).

    Args:
        image: BGR image
        model_path: Path to YOLO model
        confidence: Minimum confidence
        panel_classes: Class IDs representing panels

    Returns:
        Tuple of (contours, confidences)
    """
    detector = YOLODetector(model_path, confidence=confidence)
    panels = detector.detect_panels_only(image, panel_classes)

    contours = [p.contour for p in panels]
    confidences = [p.confidence for p in panels]

    return contours, confidences


if __name__ == '__main__':
    # Test the detector
    print("YOLO Detector Test")
    print("=" * 50)
    print(f"YOLO Available: {YOLO_AVAILABLE}")
    print()
    print("Available models:")
    for name, path in get_available_models().items():
        print(f"  - {name}: {path}")
