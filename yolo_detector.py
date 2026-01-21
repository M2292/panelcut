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


@dataclass
class SegPanel:
    """Represents a panel detected by YOLO-Seg (Instance Segmentation)"""
    mask: np.ndarray  # Segmentation mask (binary)
    polygon: np.ndarray  # Polygon points [(x1,y1), (x2,y2), ...]
    confidence: float
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # Axis-aligned bbox for reference

    @property
    def contour(self) -> np.ndarray:
        """Return polygon as OpenCV contour format for compatibility"""
        return self.polygon.reshape((-1, 1, 2)).astype(np.int32)

    @property
    def area(self) -> float:
        """Calculate area from polygon"""
        return cv2.contourArea(self.contour)

    @property
    def center(self) -> Tuple[float, float]:
        """Calculate center point"""
        M = cv2.moments(self.contour)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            return (cx, cy)
        else:
            # Fallback to bbox center
            x1, y1, x2, y2 = self.bbox
            return ((x1 + x2) / 2, (y1 + y2) / 2)


class SegmentationDetector:
    """
    YOLO-Seg based manga panel detector for arbitrary quadrilaterals.

    Uses YOLOv8-seg models that can detect instance segmentation masks,
    which allows detecting panels with arbitrary shapes (not just rotated rectangles).
    This is ideal for panels with perspective or trapezoid shapes.
    """

    DEFAULT_MODEL = 'models/manga_panels_seg_v2.pt'

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'auto'
    ):
        """
        Initialize Segmentation detector.

        Args:
            model_path: Path to segmentation YOLO model (.pt file)
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
        Load a YOLO-Seg model.

        Args:
            model_path: Path to segmentation model file

        Returns:
            True if loaded successfully
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            self.model = YOLO(model_path, task='segment')
            self.model_path = model_path
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to load segmentation model: {e}")

    def detect(self, image: np.ndarray) -> List[SegPanel]:
        """
        Run segmentation detection on an image.

        Args:
            image: BGR image (numpy array)

        Returns:
            List of SegPanel objects
        """
        if self.model is None:
            if self.model_path:
                self.load_model(self.model_path)
            else:
                raise RuntimeError("No model loaded")

        # Run inference
        results = self.model(
            image,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device if self.device != 'auto' else None
        )

        panels = []

        if len(results) == 0:
            return panels

        result = results[0]

        # Check if segmentation masks are available
        if not hasattr(result, 'masks') or result.masks is None:
            return panels

        masks = result.masks
        boxes = result.boxes

        for i in range(len(masks)):
            # Get mask polygon
            if hasattr(masks, 'xy') and masks.xy is not None:
                polygon = masks.xy[i]  # Already in xy format
            else:
                # Fallback: convert mask to polygon
                mask_data = masks.data[i].cpu().numpy()
                # Resize mask to original image size
                mask_resized = cv2.resize(
                    mask_data,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask_binary,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                if len(contours) == 0:
                    continue
                polygon = contours[0].squeeze()

            if len(polygon.shape) == 1:
                polygon = polygon.reshape(-1, 2)

            # Get bbox
            box = boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox = (int(x1), int(y1), int(x2), int(y2))

            # Get confidence and class
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names.get(class_id, f'class_{class_id}')

            # Create mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)

            panel = SegPanel(
                mask=mask,
                polygon=polygon,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                bbox=bbox
            )
            panels.append(panel)

        return panels

    def detect_panels_only(
        self,
        image: np.ndarray,
        panel_class_ids: List[int] = [0]
    ) -> List[SegPanel]:
        """
        Detect only panel objects.

        Args:
            image: BGR image
            panel_class_ids: Class IDs that represent panels

        Returns:
            List of SegPanel detections
        """
        all_detections = self.detect(image)
        return [p for p in all_detections if p.class_id in panel_class_ids]

    def create_panel_mask(
        self,
        image: np.ndarray,
        panels: List[SegPanel]
    ) -> np.ndarray:
        """
        Create a binary mask from segmentation panel detections.

        Args:
            image: Original image (for dimensions)
            panels: List of detected segmentation panels

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
        panels: List[SegPanel],
        show_confidence: bool = True,
        show_mask: bool = True
    ) -> np.ndarray:
        """
        Draw segmentation detection masks and polygons on image.

        Args:
            image: BGR image
            panels: List of detected segmentation panels
            show_confidence: Show confidence scores
            show_mask: Show semi-transparent mask overlay

        Returns:
            Image with drawn detections
        """
        viz = image.copy()
        colors = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0),
            (0, 255, 255), (255, 0, 255), (255, 255, 0),
            (128, 0, 255), (255, 128, 0)
        ]

        # Create overlay for masks
        if show_mask:
            overlay = viz.copy()

        for i, panel in enumerate(panels):
            color = colors[i % len(colors)]

            # Draw mask overlay
            if show_mask:
                overlay[panel.mask > 0] = color

            # Draw polygon outline
            polygon_int = panel.polygon.astype(np.int32)
            cv2.polylines(viz, [polygon_int], True, color, 3)

            # Build label
            label_parts = [str(i + 1)]
            if show_confidence:
                label_parts.append(f'{panel.confidence:.2f}')
            label = ' '.join(label_parts)

            # Draw label at center
            cx, cy = panel.center
            cx, cy = int(cx), int(cy)
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

        # Blend overlay with original
        if show_mask:
            viz = cv2.addWeighted(viz, 0.7, overlay, 0.3, 0)

        return viz


@dataclass
class HybridPanel:
    """Represents a panel from hybrid detection (YOLO + Seg combined)"""
    corners: np.ndarray  # 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    confidence: float
    class_id: int
    class_name: str
    source: str  # 'yolo' or 'seg' - which model this came from
    is_rectangular: bool  # True if panel is rectangular (use YOLO), False if odd-shaped (use Seg)

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

    @property
    def area(self) -> float:
        """Calculate area from corners"""
        return cv2.contourArea(self.contour)

    @property
    def center(self) -> Tuple[float, float]:
        """Calculate center point"""
        return (float(self.corners[:, 0].mean()), float(self.corners[:, 1].mean()))


class HybridDetector:
    """
    Hybrid detector that combines YOLO (for rectangular panels) and
    Segmentation (for odd-shaped/diagonal panels).

    Logic:
    1. Run both YOLO and Seg models on the image
    2. Match detections between models using IoU
    3. For each detection, check if corners form a rectangle
    4. Use YOLO detection for rectangular panels (more stable)
    5. Use Seg detection for odd-shaped panels (can capture diagonals)
    """

    def __init__(
        self,
        yolo_model_path: str = 'models/manga109_yolo.pt',
        seg_model_path: str = 'models/manga_panels_seg_v2.pt',
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        rectangularity_threshold: float = 0.02,  # Max corner deviation to consider rectangular
        device: str = 'auto'
    ):
        """
        Initialize Hybrid detector.

        Args:
            yolo_model_path: Path to YOLO model for rectangular detection
            seg_model_path: Path to Segmentation model for odd-shaped detection
            confidence: Minimum confidence threshold (0-1)
            iou_threshold: IoU threshold for matching detections
            rectangularity_threshold: Threshold for determining if a shape is rectangular
            device: Device to run on ('auto', 'cpu', 'cuda', etc.)
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "Ultralytics YOLO not installed. "
                "Install with: pip install ultralytics"
            )

        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.rectangularity_threshold = rectangularity_threshold
        self.device = device

        self.yolo_detector = None
        self.seg_detector = None
        self.yolo_model_path = yolo_model_path
        self.seg_model_path = seg_model_path

        # Load models
        self._load_models()

    def _load_models(self):
        """Load both YOLO and Segmentation models."""
        # Load YOLO model
        if os.path.exists(self.yolo_model_path):
            try:
                self.yolo_detector = YOLODetector(
                    model_path=self.yolo_model_path,
                    confidence=self.confidence,
                    iou_threshold=self.iou_threshold,
                    device=self.device
                )
                print(f"[Hybrid] Loaded YOLO model: {self.yolo_model_path}")
            except Exception as e:
                print(f"[Hybrid] Failed to load YOLO model: {e}")
        else:
            print(f"[Hybrid] YOLO model not found: {self.yolo_model_path}")

        # Load Segmentation model
        if os.path.exists(self.seg_model_path):
            try:
                self.seg_detector = SegmentationDetector(
                    model_path=self.seg_model_path,
                    confidence=self.confidence,
                    iou_threshold=self.iou_threshold,
                    device=self.device
                )
                print(f"[Hybrid] Loaded Seg model: {self.seg_model_path}")
            except Exception as e:
                print(f"[Hybrid] Failed to load Seg model: {e}")
        else:
            print(f"[Hybrid] Seg model not found: {self.seg_model_path}")

    def _corners_match_axis_aligned(self, seg_corners: np.ndarray, yolo_corners: np.ndarray, image_shape: Tuple[int, int]) -> bool:
        """
        Check if seg corners are close enough to YOLO's axis-aligned box.

        Since YOLO always outputs axis-aligned rectangles, any significant
        deviation in seg corners means seg detected something YOLO can't represent
        (diagonal edges, rotated panels, etc.)

        We are STRICT here - if there's ANY doubt, we prefer seg corners.

        Args:
            seg_corners: 4 corner points from segmentation model
            yolo_corners: 4 corner points from YOLO (always axis-aligned)
            image_shape: (height, width) for normalization

        Returns:
            True if seg corners are essentially the same as YOLO's axis-aligned box
        """
        if len(seg_corners) != 4 or len(yolo_corners) != 4:
            return False

        h, w = image_shape
        diagonal = np.sqrt(h**2 + w**2)

        # STRICT threshold: corners must be within 1% of image diagonal
        # This is about 15 pixels on a typical manga page
        # We want to be conservative - prefer seg when in doubt
        threshold = 0.01 * diagonal

        # For each seg corner, find the closest YOLO corner
        # and measure the deviation
        total_deviation = 0
        max_deviation = 0

        for seg_pt in seg_corners:
            # Find minimum distance to any YOLO corner
            min_dist = float('inf')
            for yolo_pt in yolo_corners:
                dist = np.linalg.norm(seg_pt - yolo_pt)
                min_dist = min(min_dist, dist)
            total_deviation += min_dist
            max_deviation = max(max_deviation, min_dist)

        avg_deviation = total_deviation / 4

        # Check if any edge is significantly non-horizontal/vertical
        # YOLO edges are always horizontal or vertical
        # STRICT: even 3 degrees off means it's not axis-aligned
        max_edge_angle = 0
        for i in range(4):
            p1 = seg_corners[i]
            p2 = seg_corners[(i + 1) % 4]
            edge = p2 - p1

            if np.linalg.norm(edge) > 0:
                # Angle from horizontal (0) or vertical (90)
                angle = np.degrees(np.arctan2(abs(edge[1]), abs(edge[0])))
                # Distance from nearest axis (0, 90, 180)
                angle_from_axis = min(angle, abs(90 - angle), abs(180 - angle))
                max_edge_angle = max(max_edge_angle, angle_from_axis)

        # Seg matches YOLO only if BOTH conditions are strictly met:
        # 1. All corners are very close to YOLO corners (within 1% of diagonal)
        # 2. All edges are nearly horizontal/vertical (within 3 degrees)
        corners_match = max_deviation < threshold
        edges_axis_aligned = max_edge_angle < 3

        # Debug output
        print(f"[Hybrid]   Corner deviation: max={max_deviation:.1f}px, avg={avg_deviation:.1f}px (threshold={threshold:.1f}px)")
        print(f"[Hybrid]   Edge angle from axis: {max_edge_angle:.1f}° (threshold=3°)")
        print(f"[Hybrid]   corners_match={corners_match}, edges_axis_aligned={edges_axis_aligned}")

        return corners_match and edges_axis_aligned

    def _is_rectangular(self, corners: np.ndarray, image_shape: Tuple[int, int]) -> bool:
        """
        Check if 4 corners form a rectangle (within threshold).

        Note: This is kept for backward compatibility but the hybrid detector
        now primarily uses _corners_match_axis_aligned() for better comparison.

        A rectangle has:
        - 4 corners with approximately 90-degree angles
        - Opposite sides approximately equal length

        Args:
            corners: 4 corner points as numpy array
            image_shape: (height, width) for normalization

        Returns:
            True if corners form a rectangle
        """
        if len(corners) != 4:
            return False

        h, w = image_shape
        diagonal = np.sqrt(h**2 + w**2)
        threshold = self.rectangularity_threshold * diagonal

        # Check angles at each corner (should be ~90 degrees)
        angles = []
        for i in range(4):
            p1 = corners[(i - 1) % 4]
            p2 = corners[i]
            p3 = corners[(i + 1) % 4]

            v1 = p1 - p2
            v2 = p3 - p2

            # Calculate angle using dot product
            dot = np.dot(v1, v2)
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)

            if mag1 == 0 or mag2 == 0:
                return False

            cos_angle = dot / (mag1 * mag2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            angles.append(angle)

        # All angles should be close to 90 degrees
        angle_deviation = max(abs(a - 90) for a in angles)

        # Check if opposite sides are equal length
        side_lengths = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            side_lengths.append(np.linalg.norm(p2 - p1))

        # Opposite sides should be similar
        side_diff1 = abs(side_lengths[0] - side_lengths[2]) / max(side_lengths[0], side_lengths[2], 1)
        side_diff2 = abs(side_lengths[1] - side_lengths[3]) / max(side_lengths[1], side_lengths[3], 1)

        # Rectangle if angles are close to 90 and opposite sides are similar
        is_rect = angle_deviation < 15 and side_diff1 < 0.1 and side_diff2 < 0.1

        return is_rect

    def _calculate_iou(self, corners1: np.ndarray, corners2: np.ndarray) -> float:
        """Calculate IoU between two polygons defined by corners."""
        # Create masks for each polygon
        x_min = min(corners1[:, 0].min(), corners2[:, 0].min())
        y_min = min(corners1[:, 1].min(), corners2[:, 1].min())
        x_max = max(corners1[:, 0].max(), corners2[:, 0].max())
        y_max = max(corners1[:, 1].max(), corners2[:, 1].max())

        w = int(x_max - x_min) + 1
        h = int(y_max - y_min) + 1

        if w <= 0 or h <= 0:
            return 0.0

        # Shift corners to local coordinates
        c1_local = (corners1 - [x_min, y_min]).astype(np.int32)
        c2_local = (corners2 - [x_min, y_min]).astype(np.int32)

        # Create masks
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask2 = np.zeros((h, w), dtype=np.uint8)

        cv2.fillPoly(mask1, [c1_local], 255)
        cv2.fillPoly(mask2, [c2_local], 255)

        # Calculate IoU
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def _get_corners_from_yolo(self, panel: YOLOPanel) -> np.ndarray:
        """Extract corners from YOLO panel (axis-aligned bbox)."""
        x1, y1, x2, y2 = panel.bbox
        return np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)

    def _get_corners_from_seg(self, panel: SegPanel) -> np.ndarray:
        """Extract 4 corners from segmentation panel polygon."""
        polygon = panel.polygon

        if len(polygon) == 4:
            return polygon.astype(np.float32)

        # Approximate polygon to 4 corners
        contour = panel.contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            return approx.squeeze().astype(np.float32)

        # Fallback: use bounding rect corners
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32)

    def detect(self, image: np.ndarray) -> List[HybridPanel]:
        """
        Run hybrid detection combining YOLO and Segmentation.

        Args:
            image: BGR image (numpy array)

        Returns:
            List of HybridPanel objects
        """
        h, w = image.shape[:2]
        hybrid_panels = []

        # Get detections from both models
        yolo_panels = []
        seg_panels = []

        if self.yolo_detector:
            try:
                yolo_panels = self.yolo_detector.detect_panels_only(image, panel_class_ids=[2])
                print(f"[Hybrid] YOLO detected {len(yolo_panels)} panels")
            except Exception as e:
                print(f"[Hybrid] YOLO detection failed: {e}")

        if self.seg_detector:
            try:
                seg_panels = self.seg_detector.detect_panels_only(image)
                print(f"[Hybrid] Seg detected {len(seg_panels)} panels")
            except Exception as e:
                print(f"[Hybrid] Seg detection failed: {e}")

        # If only one model is available, use its results directly
        if not seg_panels and yolo_panels:
            print("[Hybrid] Using YOLO only (no Seg results)")
            for panel in yolo_panels:
                corners = self._get_corners_from_yolo(panel)
                hybrid_panels.append(HybridPanel(
                    corners=corners,
                    confidence=panel.confidence,
                    class_id=panel.class_id,
                    class_name=panel.class_name,
                    source='yolo',
                    is_rectangular=True
                ))
            return hybrid_panels

        if not yolo_panels and seg_panels:
            print("[Hybrid] Using Seg only (no YOLO results)")
            for panel in seg_panels:
                corners = self._get_corners_from_seg(panel)
                is_rect = self._is_rectangular(corners, (h, w))
                hybrid_panels.append(HybridPanel(
                    corners=corners,
                    confidence=panel.confidence,
                    class_id=panel.class_id,
                    class_name=panel.class_name,
                    source='seg',
                    is_rectangular=is_rect
                ))
            return hybrid_panels

        if not yolo_panels and not seg_panels:
            print("[Hybrid] No detections from either model")
            return []

        # UNION APPROACH: Start with all YOLO panels, then enhance/add from seg
        # 1. Add all YOLO panels first
        # 2. For each seg panel, check if it overlaps with existing panel
        # 3. If overlap AND seg has significantly different corners (non-rect), replace with seg
        # 4. If no overlap, add seg panel as new detection

        # Start with all YOLO panels
        for yolo_panel in yolo_panels:
            yolo_corners = self._get_corners_from_yolo(yolo_panel)
            hybrid_panels.append(HybridPanel(
                corners=yolo_corners,
                confidence=yolo_panel.confidence,
                class_id=yolo_panel.class_id,
                class_name='panel',
                source='yolo',
                is_rectangular=True
            ))

        print(f"[Hybrid] Started with {len(hybrid_panels)} YOLO panels")

        # Now process seg panels - either replace overlapping YOLO or add new
        for seg_panel in seg_panels:
            seg_corners = self._get_corners_from_seg(seg_panel)

            # Find best overlapping panel in our hybrid list
            best_match_idx = -1
            best_iou = 0.3  # Minimum IoU to consider overlap

            for j, existing_panel in enumerate(hybrid_panels):
                iou = self._calculate_iou(seg_corners, existing_panel.corners)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = j

            if best_match_idx >= 0:
                # Found overlap - decide whether to replace based on corner deviation
                existing_panel = hybrid_panels[best_match_idx]
                yolo_corners = existing_panel.corners

                # KEY INSIGHT: YOLO always outputs axis-aligned rectangles
                # If seg corners differ significantly from YOLO's axis-aligned box,
                # it means seg detected diagonal/rotated edges that YOLO can't represent
                seg_matches_yolo = self._corners_match_axis_aligned(seg_corners, yolo_corners, (h, w))

                if not seg_matches_yolo:
                    # Seg corners deviate from YOLO's axis-aligned box
                    # This means seg detected diagonal/rotated/non-rectangular shape
                    # Use seg corners for better accuracy
                    hybrid_panels[best_match_idx] = HybridPanel(
                        corners=seg_corners,
                        confidence=max(seg_panel.confidence, existing_panel.confidence),
                        class_id=seg_panel.class_id,
                        class_name='panel',
                        source='seg',
                        is_rectangular=False
                    )
                    print(f"[Hybrid] Replaced YOLO with Seg (corners deviate from axis-aligned, IoU={best_iou:.2f})")
                else:
                    # Seg corners match YOLO's axis-aligned box
                    # Both agree it's a standard rectangle - keep YOLO (more stable)
                    hybrid_panels[best_match_idx] = HybridPanel(
                        corners=existing_panel.corners,
                        confidence=max(seg_panel.confidence, existing_panel.confidence),
                        class_id=existing_panel.class_id,
                        class_name='panel',
                        source='yolo',
                        is_rectangular=True
                    )
                    print(f"[Hybrid] Kept YOLO (seg matches axis-aligned, IoU={best_iou:.2f})")
            else:
                # No overlap - seg found a panel YOLO missed, add it
                is_rect = self._is_rectangular(seg_corners, (h, w))
                hybrid_panels.append(HybridPanel(
                    corners=seg_corners,
                    confidence=seg_panel.confidence,
                    class_id=seg_panel.class_id,
                    class_name='panel',
                    source='seg',
                    is_rectangular=is_rect
                ))
                print(f"[Hybrid] Added new Seg panel (no YOLO overlap)")

        print(f"[Hybrid] Final: {len(hybrid_panels)} panels "
              f"({sum(1 for p in hybrid_panels if p.source == 'yolo')} YOLO, "
              f"{sum(1 for p in hybrid_panels if p.source == 'seg')} Seg)")

        return hybrid_panels

    def detect_panels_only(
        self,
        image: np.ndarray,
        panel_class_ids: List[int] = [0, 2]
    ) -> List[HybridPanel]:
        """
        Detect only panel objects (convenience method).

        Args:
            image: BGR image
            panel_class_ids: Class IDs that represent panels

        Returns:
            List of HybridPanel detections
        """
        return self.detect(image)

    def draw_detections(
        self,
        image: np.ndarray,
        panels: List[HybridPanel],
        show_confidence: bool = True,
        show_source: bool = True
    ) -> np.ndarray:
        """
        Draw hybrid detection boxes on image.

        Args:
            image: BGR image
            panels: List of detected hybrid panels
            show_confidence: Show confidence scores
            show_source: Show source model (YOLO/Seg)

        Returns:
            Image with drawn detections
        """
        viz = image.copy()
        # Different colors for YOLO vs Seg
        yolo_color = (0, 255, 0)  # Green for YOLO
        seg_color = (0, 165, 255)  # Orange for Seg

        for i, panel in enumerate(panels):
            color = yolo_color if panel.source == 'yolo' else seg_color

            # Draw polygon
            corners_int = panel.corners.astype(np.int32)
            cv2.polylines(viz, [corners_int], True, color, 3)

            # Build label
            label_parts = [str(i + 1)]
            if show_source:
                label_parts.append(f'[{panel.source.upper()}]')
            if show_confidence:
                label_parts.append(f'{panel.confidence:.2f}')
            label = ' '.join(label_parts)

            # Draw label at center
            cx, cy = panel.center
            cx, cy = int(cx), int(cy)
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
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
