"""
Manga Panel Slicer - Core Algorithm

Hybrid detection approach:
1. YOLO-based detection (if model available) - ML-powered, high accuracy
2. Traditional CV detection - Morphological + color-based gutter detection
3. Automatic fallback between methods

Simple approach: Detect the gutter color from the image borders,
then find all pixels of that color to identify the gutter regions.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Optional YOLO import
try:
    from yolo_detector import YOLODetector, YOLOPanel, is_yolo_available, detect_panels_yolo
    YOLO_AVAILABLE = is_yolo_available()
except ImportError:
    YOLO_AVAILABLE = False
    YOLODetector = None


@dataclass
class Panel:
    """Represents a detected manga panel"""
    image: np.ndarray
    contour: np.ndarray
    bounding_box: Tuple[int, int, int, int]
    area: float
    index: int
    confidence: float = 1.0  # Detection confidence (1.0 for CV, varies for YOLO)
    class_id: int = 2  # Default: 2 (frame/panel). 0=body, 1=face, 2=frame, 3=text
    class_name: str = 'panel'  # Human-readable class name


def extract_panel_with_mask(image: np.ndarray, contour: np.ndarray,
                            transparent: bool = True) -> np.ndarray:
    """Extract a panel using its contour shape.

    Args:
        image: Source image (BGR, BGRA, or grayscale)
        contour: Panel contour
        transparent: If True, non-panel areas are transparent (BGRA output).
                    If False, non-panel areas are white (BGR output).
    """
    x, y, w, h = cv2.boundingRect(contour)
    x, y = max(0, x), max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    if w <= 0 or h <= 0:
        if transparent:
            return np.zeros((1, 1, 4), dtype=np.uint8)
        return np.zeros((1, 1, 3), dtype=np.uint8)

    # Convert BGRA to BGR if needed
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    cropped_mask = mask[y:y+h, x:x+w]
    cropped_image = image[y:y+h, x:x+w].copy()

    if transparent:
        # Output BGRA with transparent background
        if len(image.shape) == 3:
            output = np.zeros((h, w, 4), dtype=np.uint8)
            output[cropped_mask > 0, :3] = cropped_image[cropped_mask > 0]
            output[cropped_mask > 0, 3] = 255  # Alpha channel - opaque where panel exists
        else:
            # Grayscale to BGRA
            output = np.zeros((h, w, 4), dtype=np.uint8)
            output[cropped_mask > 0, 0] = cropped_image[cropped_mask > 0]
            output[cropped_mask > 0, 1] = cropped_image[cropped_mask > 0]
            output[cropped_mask > 0, 2] = cropped_image[cropped_mask > 0]
            output[cropped_mask > 0, 3] = 255
    else:
        # Output BGR with white background
        if len(image.shape) == 3:
            output = np.full((h, w, 3), 255, dtype=np.uint8)
            output[cropped_mask > 0] = cropped_image[cropped_mask > 0]
        else:
            output = np.full((h, w), 255, dtype=np.uint8)
            output[cropped_mask > 0] = cropped_image[cropped_mask > 0]

    return output


def extract_panel_bbox(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Extract panel using bounding box."""
    x, y, w, h = cv2.boundingRect(contour)
    x, y = max(0, x), max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    if w <= 0 or h <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    return image[y:y+h, x:x+w].copy()


def get_panel_center(contour: np.ndarray) -> Tuple[float, float]:
    """Get the center point of a contour."""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        return (M["m10"] / M["m00"], M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(contour)
    return (x + w / 2, y + h / 2)


def sort_panels_reading_order(panels: List[Panel], rtl: bool = True) -> List[Panel]:
    """
    Sort panels in manga reading order using row-based sorting.

    For RTL manga (Japanese style): right-to-left, top-to-bottom
    For LTR manga (Western style): left-to-right, top-to-bottom

    Algorithm:
    1. Group panels into rows based on vertical overlap
    2. Sort rows top-to-bottom
    3. Within each row, sort panels right-to-left (RTL) or left-to-right (LTR)
    """
    if not panels:
        return panels

    if len(panels) == 1:
        panels[0].index = 0
        return panels

    # Extract panel geometry with unique IDs
    panel_data = []
    for idx, panel in enumerate(panels):
        x, y, w, h = panel.bounding_box
        cx, cy = get_panel_center(panel.contour)
        panel_data.append({
            'id': idx,
            'panel': panel,
            'cx': cx,
            'cy': cy,
            'top': y,
            'bottom': y + h,
            'left': x,
            'right': x + w,
            'h': h,
            'w': w
        })

    # Group panels into rows based on vertical overlap
    # Two panels are in the same row if they have significant vertical overlap
    rows = []
    used_ids = set()

    # Sort by top position first to process top-to-bottom
    panel_data_sorted = sorted(panel_data, key=lambda p: p['top'])

    for p in panel_data_sorted:
        if p['id'] in used_ids:
            continue

        # Start a new row with this panel
        current_row = [p]
        used_ids.add(p['id'])
        row_top = p['top']
        row_bottom = p['bottom']

        # Find other panels that overlap vertically with this row
        for other in panel_data_sorted:
            if other['id'] in used_ids:
                continue

            # Check vertical overlap
            overlap_top = max(row_top, other['top'])
            overlap_bottom = min(row_bottom, other['bottom'])
            v_overlap = max(0, overlap_bottom - overlap_top)

            # Panels are in the same row if they have significant vertical overlap
            # relative to the smaller panel's height
            min_height = min(row_bottom - row_top, other['h'])
            if v_overlap > min_height * 0.5:
                current_row.append(other)
                used_ids.add(other['id'])
                # Extend row bounds
                row_top = min(row_top, other['top'])
                row_bottom = max(row_bottom, other['bottom'])

        rows.append(current_row)

    # Sort rows by their average center Y position (more robust than top edge)
    rows.sort(key=lambda row: sum(p['cy'] for p in row) / len(row))

    # Within each row, sort by horizontal position using CENTER
    # This is more robust to small bbox differences between YOLO and OBB
    sorted_panels = []
    for row in rows:
        if rtl:
            # Right-to-left: sort by center X descending
            row.sort(key=lambda p: -p['cx'])
        else:
            # Left-to-right: sort by center X ascending
            row.sort(key=lambda p: p['cx'])
        sorted_panels.extend(row)

    # Assign final indices
    result = []
    for i, p in enumerate(sorted_panels):
        p['panel'].index = i
        result.append(p['panel'])

    return result


def get_border_color(image: np.ndarray) -> np.ndarray:
    """Get the most common color from the image borders."""
    h, w = image.shape[:2]
    border = 10

    # Collect border pixels
    top = image[0:border, :].reshape(-1, 3)
    bottom = image[h-border:h, :].reshape(-1, 3)
    left = image[:, 0:border].reshape(-1, 3)
    right = image[:, w-border:w].reshape(-1, 3)

    all_border_pixels = np.vstack([top, bottom, left, right])

    # Use median for robustness
    return np.median(all_border_pixels, axis=0).astype(np.uint8)


def detect_gutter_lines(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Detect gutter lines using multiple methods:
    1. White horizontal/vertical lines (morphological)
    2. Colored regions connected to border (saturation-based)
    3. Diagonal colored lines (Hough transform)
    Returns a mask where white = gutter regions.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Method 1: Find white gutter lines (horizontal and vertical)
    _, white_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

    # Horizontal gutter detection - use smaller kernel for more sensitivity
    horiz_len = max(w // 6, 50)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    horizontal_lines = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, horizontal_kernel)

    # Vertical gutter detection
    vert_len = max(h // 6, 50)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    vertical_lines = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, vertical_kernel)

    # Method 2: Find colored gutters (high saturation = red, blue, etc.)
    saturation = hsv[:, :, 1]
    _, colored_mask = cv2.threshold(saturation, 80, 255, cv2.THRESH_BINARY)

    # Only keep colored regions connected to border
    colored_gutter = keep_border_connected_regions(colored_mask)

    # Method 3: Detect diagonal lines using edge-based approach
    # Create edge mask from colored regions
    colored_edges = cv2.Canny(colored_mask, 50, 150)

    # Also get edges from grayscale for diagonal detection
    gray_edges = cv2.Canny(gray, 50, 150)

    # Combine edges
    all_edges = cv2.bitwise_or(colored_edges, gray_edges)

    # Find lines using Hough transform with lower thresholds
    diagonal_mask = np.zeros((h, w), dtype=np.uint8)
    min_line_length = min(h, w) // 6  # Shorter minimum
    lines = cv2.HoughLinesP(all_edges, 1, np.pi/180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=30)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            if length > min_line_length:
                # Calculate angle - skip near-horizontal and near-vertical (already detected)
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                is_diagonal = 10 < angle < 80 or 100 < angle < 170

                # Sample points along the line to check if it's on colored pixels
                num_samples = max(int(length / 5), 10)
                colored_count = 0
                for i in range(num_samples):
                    t = i / max(num_samples - 1, 1)
                    px = int(x1 + t * (x2 - x1))
                    py = int(y1 + t * (y2 - y1))
                    if 0 <= px < w and 0 <= py < h:
                        if saturation[py, px] > 60:
                            colored_count += 1

                # Accept line if it's diagonal and mostly on colored pixels
                # OR if it's any orientation and highly colored
                color_ratio = colored_count / num_samples
                if (is_diagonal and color_ratio > 0.3) or color_ratio > 0.6:
                    # Draw thicker line to ensure full gutter coverage
                    cv2.line(diagonal_mask, (x1, y1), (x2, y2), 255, 20)

    # Combine all detected gutters
    gutter_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    gutter_mask = cv2.bitwise_or(gutter_mask, colored_gutter)
    gutter_mask = cv2.bitwise_or(gutter_mask, diagonal_mask)

    # Dilate to capture the full gutter width
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gutter_mask = cv2.dilate(gutter_mask, dilate_kernel, iterations=3)

    return gutter_mask


def create_gutter_mask(image: np.ndarray, gutter_color: np.ndarray, tolerance: int = 30) -> np.ndarray:
    """Create a mask of pixels matching the gutter color."""
    diff = np.abs(image.astype(np.int32) - gutter_color.astype(np.int32))
    distance = np.sum(diff, axis=2)
    mask = (distance < tolerance * 3).astype(np.uint8) * 255
    return mask


def keep_border_connected_regions(mask: np.ndarray) -> np.ndarray:
    """
    Only keep regions in the mask that are connected to the image border.
    This filters out matching colors that appear INSIDE panels.
    """
    h, w = mask.shape
    result = np.zeros_like(mask)

    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Check which components touch the border
    for label in range(1, num_labels):  # Skip background (0)
        component_mask = (labels == label).astype(np.uint8)

        # Check if this component touches any edge
        touches_top = np.any(component_mask[0, :] > 0)
        touches_bottom = np.any(component_mask[h-1, :] > 0)
        touches_left = np.any(component_mask[:, 0] > 0)
        touches_right = np.any(component_mask[:, w-1] > 0)

        if touches_top or touches_bottom or touches_left or touches_right:
            result[labels == label] = 255

    return result


def slice_manga_page(
    image: np.ndarray,
    min_area: int = 10000,
    rtl: bool = True,
    use_mask_extraction: bool = True,
    gutter_threshold: int = 240,
    method: str = "auto",
    tolerance: int = 30,
    debug: bool = False
) -> Tuple[List[Panel], np.ndarray, dict]:
    """Slice a manga page into individual panels.

    Returns: (panels, viz_image, debug_info)
    """
    # Convert BGRA to BGR if needed
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if len(image.shape) == 3:
        original = image.copy()
    else:
        original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = original.shape[:2]
    debug_info = {}

    # Step 1: Get the gutter color from borders
    gutter_color = get_border_color(original)
    debug_info['gutter_color'] = gutter_color.tolist()

    # Check if gutter is white/near-white (B&W manga)
    is_white_gutter = np.all(gutter_color > 240)
    debug_info['is_white_gutter'] = is_white_gutter

    if is_white_gutter:
        # Use line-based detection for B&W manga
        # This finds elongated white strips rather than all white pixels
        gutter_mask = detect_gutter_lines(original, debug=debug)
        if debug:
            debug_info['color_mask'] = gutter_mask.copy()
            debug_info['color_mask_cleaned'] = gutter_mask.copy()
            debug_info['gutter_mask_filtered'] = gutter_mask.copy()
    else:
        # Use color-based detection for colored gutters
        # Step 2: Create mask of all pixels matching gutter color
        color_mask = create_gutter_mask(original, gutter_color, tolerance=tolerance)
        if debug:
            debug_info['color_mask'] = color_mask.copy()

        # Step 3: Clean up small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        if debug:
            debug_info['color_mask_cleaned'] = color_mask.copy()

        # Step 4: CRITICAL - Only keep regions connected to image border
        # This removes red/white areas that are INSIDE panels
        gutter_mask = keep_border_connected_regions(color_mask)
        if debug:
            debug_info['gutter_mask_filtered'] = gutter_mask.copy()

    # Step 5: Dilate slightly to ensure panels are separated
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gutter_mask = cv2.dilate(gutter_mask, kernel, iterations=1)
    if debug:
        debug_info['gutter_mask_dilated'] = gutter_mask.copy()

    # Step 6: Invert to get panel regions
    panel_mask = cv2.bitwise_not(gutter_mask)
    if debug:
        debug_info['panel_mask'] = panel_mask.copy()

    # Step 7: Add border to close edge panels
    cv2.rectangle(panel_mask, (0, 0), (w-1, h-1), 0, 3)

    # Step 8: Find contours
    contours, _ = cv2.findContours(panel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 9: Filter and create panels
    panels = []
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, cw, ch = cv2.boundingRect(contour)
            if cw > 50 and ch > 50:
                if use_mask_extraction:
                    panel_image = extract_panel_with_mask(original, contour)
                else:
                    panel_image = extract_panel_bbox(original, contour)

                panels.append(Panel(
                    image=panel_image,
                    contour=contour,
                    bounding_box=(x, y, cw, ch),
                    area=area,
                    index=idx
                ))

    if panels:
        panels = sort_panels_reading_order(panels, rtl=rtl)

    viz_image = draw_panel_visualization(original, panels)
    return panels, viz_image, debug_info


def draw_panel_visualization(image: np.ndarray, panels: List[Panel]) -> np.ndarray:
    """Draw panel boundaries and numbers on the image.

    Each panel gets a distinct colored border and numbered label.
    """
    if not panels:
        return image.copy()

    # Convert BGRA to BGR if needed
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    viz = image.copy()
    h, w = viz.shape[:2]

    # Distinct colors for each panel (BGR format)
    colors = [
        (0, 0, 255),     # Red
        (0, 255, 0),     # Green
        (255, 0, 0),     # Blue
        (0, 255, 255),   # Yellow
        (255, 0, 255),   # Magenta
        (255, 255, 0),   # Cyan
        (0, 128, 255),   # Orange
        (128, 0, 255),   # Purple
        (0, 255, 128),   # Spring green
        (255, 128, 0),   # Sky blue
    ]

    # First pass: draw semi-transparent filled rectangles
    overlay = viz.copy()
    for i, panel in enumerate(panels):
        color = colors[i % len(colors)]
        x, y, pw, ph = panel.bounding_box
        # Ensure coordinates are integers
        x, y, pw, ph = int(x), int(y), int(pw), int(ph)
        # Draw filled rectangle on overlay
        cv2.rectangle(overlay, (x, y), (x + pw, y + ph), color, -1)

    # Blend overlay (25% fill opacity)
    cv2.addWeighted(overlay, 0.25, viz, 0.75, 0, viz)

    # Second pass: draw borders and labels on top
    for i, panel in enumerate(panels):
        color = colors[i % len(colors)]
        x, y, pw, ph = panel.bounding_box
        x, y, pw, ph = int(x), int(y), int(pw), int(ph)

        # Draw thick border (6 pixels)
        cv2.rectangle(viz, (x, y), (x + pw, y + ph), color, 6)

        # Draw panel number
        label = str(panel.index + 1)

        # Calculate font size based on panel size
        min_dim = min(pw, ph)
        font_scale = max(1.0, min(3.0, min_dim / 100))
        font_thickness = max(2, int(font_scale * 2))

        # Get text size
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # Position label in top-left corner with padding
        label_x = x + 10
        label_y = y + th + 15

        # Draw label background (solid color rectangle)
        bg_padding = 8
        cv2.rectangle(
            viz,
            (label_x - bg_padding, label_y - th - bg_padding),
            (label_x + tw + bg_padding, label_y + bg_padding),
            color, -1
        )

        # Draw white text
        cv2.putText(
            viz, label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), font_thickness, cv2.LINE_AA
        )

    return viz


def load_image(path: str) -> np.ndarray:
    """Load an image from file."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def save_panels(panels: List[Panel], output_dir: str, prefix: str = "panel") -> List[str]:
    """Save extracted panels to files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for panel in panels:
        filename = f"{prefix}_{panel.index + 1:03d}.png"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, panel.image)
        paths.append(path)

    return paths


# =============================================================================
# HYBRID DETECTION SYSTEM
# =============================================================================

def slice_manga_page_yolo(
    image: np.ndarray,
    model_path: str = "models/manga109_yolo.pt",
    confidence: float = 0.5,
    rtl: bool = True,
    use_mask_extraction: bool = True,
    min_area: int = 10000,
    panel_class_ids: Optional[List[int]] = None,
    debug: bool = False
) -> Tuple[List[Panel], np.ndarray, dict]:
    """
    Slice manga page using YOLO detection.

    Args:
        image: Input image (BGR or BGRA)
        model_path: Path to YOLO model
        confidence: Minimum detection confidence
        rtl: Right-to-left reading order
        use_mask_extraction: Use mask-based extraction
        min_area: Minimum panel area
        panel_class_ids: Which classes to detect (default: [2] for frames/panels)
                        Class 0=body, 1=face, 2=frame, 3=text
        debug: Generate debug images

    Returns:
        (panels, visualization, debug_info)
    """
    if not YOLO_AVAILABLE:
        raise ImportError("YOLO not available. Install with: pip install ultralytics")

    # Convert BGRA to BGR if needed
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    original = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = original.shape[:2]
    debug_info = {'method': 'yolo', 'model': model_path}

    try:
        detector = YOLODetector(model_path, confidence=confidence)

        # Default to class 2 (frame/panels) if not specified
        if panel_class_ids is None:
            panel_class_ids = [2]

        yolo_panels = detector.detect_panels_only(image, panel_class_ids=panel_class_ids)

        panels = []
        contours_for_mask = []
        for i, yp in enumerate(yolo_panels):
            x1, y1, x2, y2 = yp.bbox
            area = (x2 - x1) * (y2 - y1)

            if area < min_area:
                continue

            if use_mask_extraction:
                panel_image = extract_panel_with_mask(original, yp.contour)
            else:
                panel_image = extract_panel_bbox(original, yp.contour)

            panels.append(Panel(
                image=panel_image,
                contour=yp.contour,
                bounding_box=(x1, y1, x2 - x1, y2 - y1),
                area=area,
                index=i,
                confidence=yp.confidence,
                class_id=yp.class_id,
                class_name=yp.class_name
            ))
            contours_for_mask.append(yp.contour)

        print(f"[YOLO] Raw detections: {len(yolo_panels)}, after min_area filter: {len(panels)}")
        for i, yp in enumerate(yolo_panels):
            x1, y1, x2, y2 = yp.bbox
            area = (x2 - x1) * (y2 - y1)
            print(f"  Panel {i}: bbox={yp.bbox}, area={area}, conf={yp.confidence:.2f}, class={yp.class_name}")

        if panels:
            panels = sort_panels_reading_order(panels, rtl=rtl)

        debug_info['panel_count'] = len(panels)
        debug_info['confidences'] = [p.confidence for p in panels]

        # Generate debug images for frontend (mask where panels are white, gutters are black)
        if debug or True:  # Always generate for frontend compatibility
            # For YOLO, create a mask by drawing panel OUTLINES (not filled)
            # This creates visible gutter lines between panels instead of overlapping fills
            print(f"[YOLO] Creating mask with {len(contours_for_mask)} contours on image size {w}x{h}")

            # Start with white background (everything is "panel" by default)
            panel_mask = np.ones((h, w), dtype=np.uint8) * 255

            # Draw black borders around each panel to create gutter lines
            border_width = 15  # Width of gutter lines
            for i, contour in enumerate(contours_for_mask):
                bbox = cv2.boundingRect(contour)
                print(f"  Contour {i}: shape={contour.shape}, bounds={bbox}")
                # Draw thick black outline around each panel
                cv2.drawContours(panel_mask, [contour], -1, 0, border_width)

            # Also draw black borders at the image edges
            cv2.rectangle(panel_mask, (0, 0), (w-1, h-1), 0, border_width)

            # For the gutter mask (inverted) - white = gutter, black = panel
            gutter_mask = cv2.bitwise_not(panel_mask)

            # Encode masks as base64 for frontend
            _, panel_mask_encoded = cv2.imencode('.png', panel_mask)
            _, gutter_mask_encoded = cv2.imencode('.png', gutter_mask)

            import base64
            debug_info['debug_images'] = {
                'panel_mask': base64.b64encode(panel_mask_encoded).decode('utf-8'),
                'gutter_mask_dilated': base64.b64encode(gutter_mask_encoded).decode('utf-8'),
                'color_mask': base64.b64encode(gutter_mask_encoded).decode('utf-8'),
            }
            debug_info['gutter_color'] = [255, 255, 255]  # Placeholder for YOLO

        print(f"[YOLO] Drawing visualization for {len(panels)} panels:")
        for p in panels:
            print(f"  Panel {p.index}: bbox={p.bounding_box}")
        viz_image = draw_panel_visualization(original, panels)
        return panels, viz_image, debug_info

    except Exception as e:
        debug_info['error'] = str(e)
        import traceback
        traceback.print_exc()
        return [], original.copy(), debug_info


def slice_manga_page_hybrid(
    image: np.ndarray,
    min_area: int = 10000,
    rtl: bool = True,
    use_mask_extraction: bool = True,
    gutter_threshold: int = 240,
    tolerance: int = 30,
    yolo_model: Optional[str] = None,
    yolo_confidence: float = 0.5,
    yolo_classes: Optional[List[int]] = None,
    prefer_yolo: bool = True,
    fallback_on_failure: bool = True,
    min_panels_threshold: int = 2,
    debug: bool = False
) -> Tuple[List[Panel], np.ndarray, dict]:
    """
    Hybrid detection combining YOLO and traditional CV.

    Strategy:
    1. If YOLO model available and prefer_yolo=True, try YOLO first
    2. If YOLO fails or finds too few panels, fall back to CV
    3. Optionally merge results from both methods

    Args:
        image: Input image (BGR or BGRA)
        min_area: Minimum panel area
        rtl: Right-to-left reading order
        use_mask_extraction: Use mask-based extraction
        gutter_threshold: Threshold for white gutter detection
        tolerance: Color tolerance for gutter detection
        yolo_model: Path to YOLO model (None to skip YOLO)
        yolo_confidence: YOLO confidence threshold
        yolo_classes: Which YOLO classes to detect (default: [2] for frames)
                     Class 0=body, 1=face, 2=frame, 3=text
        prefer_yolo: Try YOLO first if available
        fallback_on_failure: Fall back to CV if YOLO fails
        min_panels_threshold: Minimum panels for YOLO result to be accepted
        debug: Include debug information

    Returns:
        (panels, visualization, debug_info)
    """
    # Convert BGRA to BGR if needed
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    debug_info = {
        'hybrid': True,
        'yolo_available': YOLO_AVAILABLE,
        'yolo_model': yolo_model,
        'method_used': None
    }

    yolo_result = None
    cv_result = None

    # Try YOLO first if preferred and available
    if prefer_yolo and YOLO_AVAILABLE and yolo_model:
        try:
            yolo_panels, yolo_viz, yolo_debug = slice_manga_page_yolo(
                image,
                model_path=yolo_model,
                confidence=yolo_confidence,
                rtl=rtl,
                use_mask_extraction=use_mask_extraction,
                min_area=min_area,
                panel_class_ids=yolo_classes
            )

            if len(yolo_panels) >= min_panels_threshold:
                debug_info['method_used'] = 'yolo'
                debug_info['yolo_debug'] = yolo_debug
                # Copy debug_images to top level for frontend compatibility
                if 'debug_images' in yolo_debug:
                    debug_info['debug_images'] = yolo_debug['debug_images']
                if 'gutter_color' in yolo_debug:
                    debug_info['gutter_color'] = yolo_debug['gutter_color']
                return yolo_panels, yolo_viz, debug_info

            yolo_result = (yolo_panels, yolo_viz, yolo_debug)
            debug_info['yolo_panel_count'] = len(yolo_panels)
            debug_info['yolo_below_threshold'] = True

        except Exception as e:
            debug_info['yolo_error'] = str(e)

    # Fall back to or use CV detection
    if fallback_on_failure or not prefer_yolo or yolo_result is None:
        cv_panels, cv_viz, cv_debug = slice_manga_page(
            image,
            min_area=min_area,
            rtl=rtl,
            use_mask_extraction=use_mask_extraction,
            gutter_threshold=gutter_threshold,
            tolerance=tolerance,
            debug=debug
        )
        cv_result = (cv_panels, cv_viz, cv_debug)
        debug_info['cv_panel_count'] = len(cv_panels)

    # Decide which result to use
    if yolo_result and cv_result:
        yolo_count = len(yolo_result[0])
        cv_count = len(cv_result[0])

        # Use YOLO if it found more panels (usually more accurate)
        if yolo_count >= cv_count and yolo_count >= min_panels_threshold:
            debug_info['method_used'] = 'yolo'
            debug_info['decision'] = f'yolo ({yolo_count}) >= cv ({cv_count})'
            return yolo_result
        else:
            debug_info['method_used'] = 'cv'
            debug_info['decision'] = f'cv ({cv_count}) > yolo ({yolo_count})'
            return cv_result

    elif yolo_result:
        debug_info['method_used'] = 'yolo'
        return yolo_result

    else:
        debug_info['method_used'] = 'cv'
        return cv_result


def get_detection_capabilities() -> Dict[str, Any]:
    """
    Get available detection capabilities.

    Returns:
        Dictionary of available features
    """
    capabilities = {
        'cv_detection': True,
        'yolo_detection': YOLO_AVAILABLE,
        'hybrid_detection': YOLO_AVAILABLE,
        'methods': ['cv']
    }

    if YOLO_AVAILABLE:
        capabilities['methods'].extend(['yolo', 'hybrid'])

        # Check for available models
        import os
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        if os.path.exists(models_dir):
            capabilities['available_models'] = [
                f for f in os.listdir(models_dir) if f.endswith('.pt')
            ]
        else:
            capabilities['available_models'] = []

    return capabilities


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python panel_slicer.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_panels"

    print(f"Loading image: {image_path}")
    image = load_image(image_path)

    print("Slicing panels...")
    panels, viz = slice_manga_page(image)

    print(f"Found {len(panels)} panels")

    cv2.imwrite("panel_detection.png", viz)
    print("Saved visualization to panel_detection.png")

    paths = save_panels(panels, output_dir)
    for path in paths:
        print(f"Saved: {path}")
