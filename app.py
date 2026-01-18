"""
Manga Panel Slicer - Web Application
Flask server with REST API for panel slicing

Features:
- AI-powered panel detection
- Hybrid detection with fallback
- Vetting workflow for panel review
- PSD export for Photoshop integration
"""

import os
import io
import base64
import uuid
import zipfile
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from panel_slicer import (
    slice_manga_page,
    Panel,
    get_detection_capabilities,
    slice_manga_page_hybrid,
    YOLO_AVAILABLE
)
from typing import List, Dict, Any

# Import optional modules
try:
    from vetting import VettingWorkflow, PanelStatus, quick_vet
    VETTING_AVAILABLE = True
except ImportError:
    VETTING_AVAILABLE = False

try:
    from psd_export import (
        export_panels_as_psd,
        panels_to_psd_bytes,
        is_psd_export_available
    )
    PSD_AVAILABLE = True
except ImportError:
    PSD_AVAILABLE = False

# Try to import rate limiter (optional for local dev)
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    LIMITER_AVAILABLE = True
except ImportError:
    LIMITER_AVAILABLE = False

app = Flask(__name__, static_folder='static')

# =============================================================================
# CONFIGURATION
# =============================================================================
# Production settings - override with environment variables
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_UPLOAD_SIZE', 15 * 1024 * 1024))  # 15MB default
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# CORS - restrict in production
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*').split(',')
if ALLOWED_ORIGINS == ['*']:
    CORS(app)
else:
    CORS(app, origins=ALLOWED_ORIGINS)

# Rate limiting (if available)
if LIMITER_AVAILABLE:
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri=os.environ.get('REDIS_URL', 'memory://')
    )
else:
    # Dummy limiter decorator that does nothing
    class DummyLimiter:
        def limit(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
    limiter = DummyLimiter()

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('static', 'index.html')


@app.route('/api/slice', methods=['POST'])
@limiter.limit("10 per minute")
def slice_image():
    """
    Slice a manga page into panels

    Accepts:
        - multipart/form-data with 'image' file
        - JSON with 'image' as base64 string

    Query params:
        - min_area: Minimum contour area (default: 10000)
        - rtl: Reading order right-to-left (default: true)
        - use_mask: Use contour mask for extraction (default: true)
        - gutter_threshold: Brightness threshold for gutter detection (default: 240)
        - method: Detection method - "auto", "flood", "contour", "hybrid" (default: auto)
        - tolerance: Color tolerance for gutter detection (default: 30)
        - debug: Return debug images (default: false)

    Returns:
        JSON with visualization and extracted panels as base64
    """
    try:
        # Get parameters
        min_area = request.args.get('min_area', 10000, type=int)
        rtl = request.args.get('rtl', 'true').lower() == 'true'
        use_mask = request.args.get('use_mask', 'true').lower() == 'true'
        gutter_threshold = request.args.get('gutter_threshold', 240, type=int)
        method = request.args.get('method', 'auto')
        tolerance = request.args.get('tolerance', 30, type=int)
        debug = request.args.get('debug', 'false').lower() == 'true'

        # Get image from request
        if request.content_type and 'multipart/form-data' in request.content_type:
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            # Read image from file
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400

            image = base64_to_image(data['image'])
        else:
            return jsonify({'error': 'Invalid content type'}), 400

        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Slice the manga page
        panels, viz_image, debug_info = slice_manga_page(
            image,
            min_area=min_area,
            rtl=rtl,
            use_mask_extraction=use_mask,
            gutter_threshold=gutter_threshold,
            method=method,
            tolerance=tolerance,
            debug=debug
        )

        # Convert results to base64
        viz_base64 = image_to_base64(viz_image)
        panels_data = []

        for panel in panels:
            panel_data = {
                'index': panel.index,
                'image': image_to_base64(panel.image),
                'width': panel.image.shape[1],
                'height': panel.image.shape[0],
                'area': panel.area
            }
            panels_data.append(panel_data)

        response_data = {
            'success': True,
            'visualization': viz_base64,
            'panels': panels_data,
            'panel_count': len(panels),
            'gutter_color': debug_info.get('gutter_color', [])
        }

        # Add debug images if requested
        if debug:
            debug_images = {}
            for key in ['color_mask', 'color_mask_cleaned', 'gutter_mask_filtered', 'gutter_mask_dilated', 'panel_mask']:
                if key in debug_info:
                    debug_images[key] = image_to_base64(debug_info[key])
            response_data['debug_images'] = debug_images

        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect_mask', methods=['POST'])
def detect_mask():
    """
    Detect panel boundaries and return an editable mask.
    Uses the same algorithm as slice_manga_page for consistency.
    Returns a black and white mask where black = divider lines (gutters), white = panels.
    """
    try:
        from panel_slicer import get_border_color, create_gutter_mask, keep_border_connected_regions, detect_gutter_lines

        min_area = request.args.get('min_area', 10000, type=int)
        rtl = request.args.get('rtl', 'true').lower() == 'true'
        tolerance = request.args.get('tolerance', 30, type=int)

        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        h, w = image.shape[:2]

        # Use the same algorithm as slice_manga_page
        gutter_color = get_border_color(image)
        is_white_gutter = np.all(gutter_color > 240)
        print(f"[detect_mask] Detected gutter color (BGR): {gutter_color}, is_white={is_white_gutter}")

        if is_white_gutter:
            # Use line-based detection for B&W manga
            gutter_mask = detect_gutter_lines(image)
        else:
            # Use color-based detection for colored gutters
            color_mask = create_gutter_mask(image, gutter_color, tolerance=tolerance)

            # Clean up small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

            # Only keep regions connected to image border
            gutter_mask = keep_border_connected_regions(color_mask)

        # Dilate slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gutter_mask = cv2.dilate(gutter_mask, kernel, iterations=1)

        # Invert to get panel mask (white = panels, black = gutters)
        panel_mask = cv2.bitwise_not(gutter_mask)

        # Add border as black (divider) to ensure edge panels are bounded
        cv2.rectangle(panel_mask, (0, 0), (w-1, h-1), 0, 3)

        # Count panels
        panel_contours, _ = cv2.findContours(panel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        panel_count = sum(1 for c in panel_contours if cv2.contourArea(c) > min_area)

        print(f"[detect_mask] Found {panel_count} panels with tolerance={tolerance}")

        # Convert mask to base64
        _, buffer = cv2.imencode('.png', panel_mask)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'mask': mask_base64,
            'panel_count': panel_count,
            'gutter_color': gutter_color.tolist(),
            'is_white_gutter': bool(is_white_gutter)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/slice_with_mask', methods=['POST'])
def slice_with_mask():
    """
    Slice a manga page using a user-provided mask image.
    The mask should have black lines where panel dividers are.
    """
    try:
        min_area = request.args.get('min_area', 10000, type=int)
        rtl = request.args.get('rtl', 'true').lower() == 'true'

        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({'error': 'Both image and mask files required'}), 400

        # Read original image
        image_file = request.files['image']
        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        # Read mask image
        mask_file = request.files['mask']
        mask_bytes = np.frombuffer(mask_file.read(), np.uint8)
        mask = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            return jsonify({'error': 'Could not decode images'}), 400

        # Resize mask to match image if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # The mask has: white (255) = panels, black (0) = gutters/dividers
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours of panel regions (white areas in the mask)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Import Panel class
        from panel_slicer import Panel, extract_panel_with_mask, sort_panels_reading_order

        panels = []
        panel_idx = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                panel_image = extract_panel_with_mask(image, contour)

                panel = Panel(
                    image=panel_image,
                    contour=contour,
                    bounding_box=(x, y, w, h),
                    area=area,
                    index=panel_idx
                )
                panels.append(panel)
                panel_idx += 1

        # Sort panels in reading order
        if panels:
            panels = sort_panels_reading_order(panels, rtl=rtl)

        # Create visualization
        viz_image = image.copy()
        colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0)]
        for panel in panels:
            color = colors[panel.index % len(colors)]
            cv2.drawContours(viz_image, [panel.contour], -1, color, 3)
            x, y, w, h = panel.bounding_box
            cv2.putText(viz_image, str(panel.index + 1), (x + 10, y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Convert to response
        viz_base64 = image_to_base64(viz_image)
        panels_data = []
        for panel in panels:
            panels_data.append({
                'index': panel.index,
                'image': image_to_base64(panel.image),
                'width': panel.image.shape[1],
                'height': panel.image.shape[0],
                'area': panel.area
            })

        return jsonify({
            'success': True,
            'visualization': viz_base64,
            'panels': panels_data,
            'panel_count': len(panels)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/slice_with_boxes', methods=['POST'])
def slice_with_boxes():
    """
    Slice a manga page using user-provided bounding boxes directly.
    This enables dynamic editing without requiring a commit step.

    Expects:
        - multipart/form-data with 'image' file
        - 'boxes' field containing JSON array of bounding boxes

    Each box should have: {x, y, width, height, index}
    """
    try:
        from panel_slicer import Panel, sort_panels_reading_order

        min_area = request.args.get('min_area', 10000, type=int)
        rtl = request.args.get('rtl', 'true').lower() == 'true'

        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        if 'boxes' not in request.form:
            return jsonify({'error': 'No bounding boxes provided'}), 400

        # Read original image
        image_file = request.files['image']
        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Parse bounding boxes
        import json
        boxes = json.loads(request.form['boxes'])
        print(f"[slice_with_boxes] Received {len(boxes)} boxes: {boxes}")

        panels = []
        for box in boxes:
            x = int(box['x'])
            y = int(box['y'])
            w = int(box['width'])
            h = int(box['height'])
            idx = int(box.get('index', len(panels)))

            # Clamp to image bounds
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)

            area = w * h
            if area < min_area:
                continue

            # Extract panel region
            panel_image = image[y:y+h, x:x+w].copy()

            # Create simple rectangular contour
            contour = np.array([
                [[x, y]],
                [[x + w, y]],
                [[x + w, y + h]],
                [[x, y + h]]
            ], dtype=np.int32)

            panel = Panel(
                image=panel_image,
                contour=contour,
                bounding_box=(x, y, w, h),
                area=area,
                index=idx
            )
            panels.append(panel)

        # Sort panels in reading order
        if panels:
            panels = sort_panels_reading_order(panels, rtl=rtl)

        # Create visualization
        from panel_slicer import draw_panel_visualization
        viz_image = draw_panel_visualization(image, panels)

        # Convert to response
        viz_base64 = image_to_base64(viz_image)
        panels_data = []
        for panel in panels:
            x, y, w, h = panel.bounding_box
            panels_data.append({
                'index': int(panel.index),
                'image': image_to_base64(panel.image),
                'width': int(panel.image.shape[1]),
                'height': int(panel.image.shape[0]),
                'area': int(panel.area),
                'bounding_box': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                }
            })

        return jsonify({
            'success': True,
            'visualization': viz_base64,
            'panels': panels_data,
            'panel_count': len(panels),
            'detection_method': 'manual'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/download', methods=['POST'])
@limiter.limit("20 per minute")
def download_panels():
    """
    Download panels as a ZIP file

    Accepts JSON with:
        - panels: Array containing base64 images
        - original_image (optional): Base64 of original manga page for training
        - boxes (optional): Bounding boxes for training data collection
    """
    try:
        data = request.get_json()
        if not data or 'panels' not in data:
            return jsonify({'error': 'No panels provided'}), 400

        panels = data['panels']

        # Silently save training data if original image and boxes are provided
        original_image = data.get('original_image')
        boxes = data.get('boxes')
        if original_image and boxes:
            try:
                _save_training_data_silent(original_image, boxes)
            except Exception as e:
                # Don't fail the download if training data save fails
                print(f"[Training] Failed to save training data: {e}")

        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for panel in panels:
                idx = panel.get('index', 0)
                img_base64 = panel.get('image', '')

                # Decode base64 to image bytes
                if ',' in img_base64:
                    img_base64 = img_base64.split(',')[1]
                img_bytes = base64.b64decode(img_base64)

                # Add to ZIP
                filename = f"panel_{idx + 1:03d}.png"
                zip_file.writestr(filename, img_bytes)

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='manga_panels.zip'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _save_training_data_silent(original_image_base64: str, boxes: list):
    """
    Silently save training data for OBB model training.
    Called when users download panels to collect training samples.
    """
    # Decode original image
    if ',' in original_image_base64:
        original_image_base64 = original_image_base64.split(',')[1]

    img_data = base64.b64decode(original_image_base64)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return

    img_height, img_width = image.shape[:2]

    # Generate unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sample_id = str(uuid.uuid4())[:8]
    image_filename = f"manga_{timestamp}_{sample_id}.png"
    label_filename = f"manga_{timestamp}_{sample_id}.txt"

    # Create OBB directories
    obb_images_dir = os.path.join(TRAINING_DATA_FOLDER, 'obb', 'images')
    obb_labels_dir = os.path.join(TRAINING_DATA_FOLDER, 'obb', 'labels')
    os.makedirs(obb_images_dir, exist_ok=True)
    os.makedirs(obb_labels_dir, exist_ok=True)

    # Save image
    cv2.imwrite(os.path.join(obb_images_dir, image_filename), image)

    # Convert boxes to OBB format
    obb_labels = []
    class_id = 0  # Single class: "panel"

    for box in boxes:
        # Check if box has corners (polygon) or just x,y,width,height
        if 'corners' in box and len(box['corners']) >= 4:
            corners = box['corners'][:4]
        else:
            # Standard bbox format - create 4 corners
            x = float(box.get('x', 0))
            y = float(box.get('y', 0))
            w = float(box.get('width', 0))
            h = float(box.get('height', 0))
            corners = [
                {'x': x, 'y': y},
                {'x': x + w, 'y': y},
                {'x': x + w, 'y': y + h},
                {'x': x, 'y': y + h}
            ]

        # Normalize and format for OBB
        obb_parts = [str(class_id)]
        for corner in corners:
            nx = max(0, min(1, corner['x'] / img_width))
            ny = max(0, min(1, corner['y'] / img_height))
            obb_parts.append(f"{nx:.6f}")
            obb_parts.append(f"{ny:.6f}")
        obb_labels.append(' '.join(obb_parts))

    # Save labels
    if obb_labels:
        with open(os.path.join(obb_labels_dir, label_filename), 'w') as f:
            f.write('\n'.join(obb_labels))

        print(f"[Training] Silently saved: {image_filename} with {len(boxes)} panels")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Manga Panel Slicer is running'})


@app.route('/api/download_bulk', methods=['POST'])
def download_bulk_panels():
    """
    Download panels from multiple pages as a ZIP file

    Expects JSON:
        - pages: Array of page arrays, each containing panel objects

    Naming: page_001_panel_001.png, page_001_panel_002.png, etc.
    """
    try:
        data = request.get_json()
        if not data or 'pages' not in data:
            return jsonify({'error': 'No pages provided'}), 400

        pages = data['pages']

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for page_idx, panels in enumerate(pages):
                if not panels:
                    continue
                for panel in panels:
                    panel_idx = panel.get('index', 0)
                    img_base64 = panel.get('image', '')

                    if ',' in img_base64:
                        img_base64 = img_base64.split(',')[1]
                    img_bytes = base64.b64decode(img_base64)

                    # page_001_panel_001.png format
                    filename = f"page_{page_idx + 1:03d}_panel_{panel_idx + 1:03d}.png"
                    zip_file.writestr(filename, img_bytes)

        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='manga_panels_bulk.zip'
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =============================================================================
# NEW ENDPOINTS: Capabilities, YOLO, Vetting, PSD Export
# =============================================================================

@app.route('/api/capabilities', methods=['GET'])
def get_capabilities():
    """
    Get available detection and export capabilities.

    Returns:
        JSON with available features
    """
    capabilities = get_detection_capabilities()
    capabilities['vetting'] = VETTING_AVAILABLE
    capabilities['psd_export'] = PSD_AVAILABLE
    capabilities['psd_capabilities'] = is_psd_export_available() if PSD_AVAILABLE else {}

    # In production, hide model details from response
    is_production = os.environ.get('FLASK_ENV') == 'production' or os.environ.get('RENDER')
    if is_production:
        # Remove sensitive model info in production
        capabilities.pop('available_models', None)
        capabilities.pop('model_path', None)

    return jsonify(capabilities)


@app.route('/api/slice_hybrid', methods=['POST'])
@limiter.limit("10 per minute")
def slice_hybrid():
    """
    Slice manga page using hybrid detection (YOLO + CV fallback).

    Query params:
        - All params from /api/slice
        - yolo_model: Path to YOLO model (optional)
        - yolo_confidence: YOLO confidence threshold (default: 0.5)
        - prefer_yolo: Try YOLO first (default: true)

    Returns:
        Same format as /api/slice with additional detection method info
    """
    try:
        # Standard params
        min_area = request.args.get('min_area', 10000, type=int)
        rtl = request.args.get('rtl', 'true').lower() == 'true'
        use_mask = request.args.get('use_mask', 'true').lower() == 'true'
        gutter_threshold = request.args.get('gutter_threshold', 240, type=int)
        tolerance = request.args.get('tolerance', 30, type=int)
        debug = request.args.get('debug', 'false').lower() == 'true'

        # Hybrid-specific params
        yolo_model = request.args.get('yolo_model', None)
        yolo_confidence = request.args.get('yolo_confidence', 0.5, type=float)
        prefer_yolo = request.args.get('prefer_yolo', 'true').lower() == 'true'

        # Get image
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Use hybrid detection
        panels, viz_image, debug_info = slice_manga_page_hybrid(
            image,
            min_area=min_area,
            rtl=rtl,
            use_mask_extraction=use_mask,
            gutter_threshold=gutter_threshold,
            tolerance=tolerance,
            yolo_model=yolo_model,
            yolo_confidence=yolo_confidence,
            prefer_yolo=prefer_yolo,
            debug=debug
        )

        # Build response
        viz_base64 = image_to_base64(viz_image)
        panels_data = []
        for panel in panels:
            x, y, w, h = panel.bounding_box
            panels_data.append({
                'index': int(panel.index),
                'image': image_to_base64(panel.image),
                'width': int(panel.image.shape[1]),
                'height': int(panel.image.shape[0]),
                'area': int(panel.area),
                'confidence': float(panel.confidence),
                'bounding_box': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                }
            })

        # Convert numpy types in debug_info to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        clean_debug_info = convert_numpy_types(debug_info)

        # Build response with debug_images at top level for frontend compatibility
        response = {
            'success': True,
            'visualization': viz_base64,
            'panels': panels_data,
            'panel_count': len(panels),
            'detection_method': debug_info.get('method_used', 'cv'),
            'debug_info': clean_debug_info if debug else {}
        }

        # Always include debug_images and gutter_color for mask preview (even when debug=false)
        if 'debug_images' in debug_info:
            response['debug_images'] = debug_info['debug_images']
        if 'gutter_color' in debug_info:
            response['gutter_color'] = debug_info['gutter_color']

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/slice_obb', methods=['POST'])
@limiter.limit("10 per minute")
def slice_obb():
    """
    Slice manga page using OBB (Oriented Bounding Box) detection.
    This endpoint uses a trained OBB model to detect diagonal/rotated panels.

    Query params:
        - min_area: Minimum panel area (default: 10000)
        - rtl: Reading order right-to-left (default: true)
        - obb_model: Path to OBB model (default: models/manga_panels_obb.pt)
        - obb_confidence: Confidence threshold (default: 0.5)

    Returns:
        JSON with panels including corner coordinates for rotated boxes
    """
    try:
        from panel_slicer import Panel, sort_panels_reading_order, extract_panel_with_mask

        min_area = request.args.get('min_area', 10000, type=int)
        rtl = request.args.get('rtl', 'true').lower() == 'true'
        obb_model = request.args.get('obb_model', 'models/manga_panels_obb.pt')
        obb_confidence = request.args.get('obb_confidence', 0.5, type=float)

        # Get image
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Check if OBB model exists
        if not os.path.exists(obb_model):
            return jsonify({
                'error': f'OBB model not found at {obb_model}. Train an OBB model first using /api/finetune_obb',
                'model_exists': False
            }), 400

        # Import and use OBB detector
        try:
            from yolo_detector import OBBDetector
        except ImportError:
            return jsonify({'error': 'OBB detector not available'}), 500

        detector = OBBDetector(model_path=obb_model, confidence=obb_confidence)
        obb_panels = detector.detect_panels_only(image)

        if not obb_panels:
            return jsonify({
                'success': True,
                'panels': [],
                'panel_count': 0,
                'detection_method': 'obb',
                'message': 'No panels detected by OBB model'
            })

        # Convert OBB panels to standard Panel format for extraction
        panels = []
        for i, obb_panel in enumerate(obb_panels):
            if obb_panel.size[0] * obb_panel.size[1] < min_area:
                continue

            # Extract panel using the rotated contour
            panel_image = extract_panel_with_mask(image, obb_panel.contour)

            # Get axis-aligned bbox for compatibility
            x1, y1, x2, y2 = obb_panel.bbox
            bbox = (x1, y1, x2 - x1, y2 - y1)

            panel = Panel(
                image=panel_image,
                contour=obb_panel.contour,
                bounding_box=bbox,
                area=obb_panel.size[0] * obb_panel.size[1],
                index=i,
                confidence=obb_panel.confidence
            )
            panels.append(panel)

        # Sort panels in reading order
        if panels:
            panels = sort_panels_reading_order(panels, rtl=rtl)

        # Create visualization with OBB boxes
        viz_image = detector.draw_detections(image, obb_panels)

        # Build response
        viz_base64 = image_to_base64(viz_image)
        panels_data = []
        for panel in panels:
            x, y, w, h = panel.bounding_box
            # Find corresponding OBB panel to get corners
            obb_panel = obb_panels[panel.index] if panel.index < len(obb_panels) else None

            panel_data = {
                'index': int(panel.index),
                'image': image_to_base64(panel.image),
                'width': int(panel.image.shape[1]),
                'height': int(panel.image.shape[0]),
                'area': int(panel.area),
                'confidence': float(panel.confidence),
                'bounding_box': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                }
            }

            # Add OBB-specific data (corners, angle)
            if obb_panel is not None:
                panel_data['corners'] = [
                    {'x': float(c[0]), 'y': float(c[1])}
                    for c in obb_panel.corners
                ]
                panel_data['angle'] = float(obb_panel.angle)
                panel_data['is_rotated'] = bool(abs(obb_panel.angle) > 2)

            panels_data.append(panel_data)

        return jsonify({
            'success': True,
            'visualization': viz_base64,
            'panels': panels_data,
            'panel_count': len(panels),
            'detection_method': 'obb',
            'model_used': obb_model
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Vetting workflow endpoints
if VETTING_AVAILABLE:
    vetting_workflow = VettingWorkflow()

    @app.route('/api/vetting/create', methods=['POST'])
    def create_vetting_session():
        """
        Create a new vetting session from detected panels.

        Expects JSON:
            - source_image: Image identifier
            - panels: Array of panel data
            - detection_method: Method used ("cv", "yolo", "hybrid")

        Returns:
            Session info
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            source_image = data.get('source_image', 'unknown')
            panels = data.get('panels', [])
            detection_method = data.get('detection_method', 'cv')
            detection_params = data.get('detection_params', {})

            session = vetting_workflow.create_session(
                source_image=source_image,
                panels_data=panels,
                detection_method=detection_method,
                detection_params=detection_params
            )

            return jsonify({
                'success': True,
                'session': session.to_dict()
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/vetting/session/<session_id>', methods=['GET'])
    def get_vetting_session(session_id):
        """Get vetting session details"""
        try:
            session = vetting_workflow.load_session(session_id)
            return jsonify({
                'success': True,
                'session': session.to_dict()
            })
        except FileNotFoundError:
            return jsonify({'error': 'Session not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/vetting/session/<session_id>/status', methods=['POST'])
    def update_panel_status(session_id):
        """
        Update status of a panel in the vetting session.

        Expects JSON:
            - panel_index: Index of panel
            - status: New status (approved, rejected, needs_edit, needs_photoshop)
            - notes: Optional notes
        """
        try:
            vetting_workflow.load_session(session_id)
            data = request.get_json()

            panel_index = data.get('panel_index')
            status_str = data.get('status')
            notes = data.get('notes', '')

            status = PanelStatus(status_str)
            vetting_workflow.set_panel_status(panel_index, status, notes)
            vetting_workflow.save_session()

            return jsonify({
                'success': True,
                'stats': vetting_workflow.get_session_stats()
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/vetting/session/<session_id>/approve_all', methods=['POST'])
    def approve_all_panels(session_id):
        """Approve all pending panels"""
        try:
            vetting_workflow.load_session(session_id)
            vetting_workflow.approve_all_pending()
            vetting_workflow.save_session()

            return jsonify({
                'success': True,
                'stats': vetting_workflow.get_session_stats()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/vetting/session/<session_id>/export', methods=['POST'])
    def export_approved_panels(session_id):
        """
        Export approved panels from vetting session.

        Query params:
            - format: Output format (png, jpg, psd)
        """
        try:
            vetting_workflow.load_session(session_id)
            output_format = request.args.get('format', 'png')

            if output_format == 'psd' and PSD_AVAILABLE:
                approved = vetting_workflow.current_session.get_approved_panels()
                panels_data = [{'image': p.image_data, 'index': p.index} for p in approved]
                psd_bytes = panels_to_psd_bytes(panels_data)

                return send_file(
                    io.BytesIO(psd_bytes),
                    mimetype='application/octet-stream',
                    as_attachment=True,
                    download_name=f'panels_{session_id}.psd'
                )
            else:
                # Export as ZIP
                output_dir = os.path.join(OUTPUT_FOLDER, session_id)
                exported = vetting_workflow.export_approved(output_dir)

                # Create ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zf:
                    for path in exported:
                        zf.write(path, os.path.basename(path))

                zip_buffer.seek(0)
                return send_file(
                    zip_buffer,
                    mimetype='application/zip',
                    as_attachment=True,
                    download_name=f'panels_{session_id}.zip'
                )

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/vetting/sessions', methods=['GET'])
    def list_vetting_sessions():
        """List all vetting sessions"""
        try:
            sessions = vetting_workflow.list_sessions()
            return jsonify({
                'success': True,
                'sessions': sessions
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/vetting/open_editor', methods=['POST'])
    def open_in_editor():
        """
        Open panel in external editor.

        Expects JSON:
            - session_id: Session ID
            - panel_index: Panel index
            - editor: Editor type (photoshop, default)
        """
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            panel_index = data.get('panel_index')
            editor = data.get('editor', 'default')

            vetting_workflow.load_session(session_id)

            if editor == 'photoshop':
                path = vetting_workflow.open_in_photoshop(panel_index)
            else:
                path = vetting_workflow.open_in_default_editor(panel_index)

            if path:
                vetting_workflow.save_session()
                return jsonify({
                    'success': True,
                    'temp_path': path,
                    'message': 'Panel opened in editor'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Could not open editor'
                }), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500


# PSD Export endpoints
if PSD_AVAILABLE:
    @app.route('/api/export/psd', methods=['POST'])
    def export_psd():
        """
        Export panels as PSD file.

        Expects JSON:
            - panels: Array of panel data with base64 images
            - layout: Layout mode (stacked, grid, original)

        Returns:
            PSD file download
        """
        try:
            data = request.get_json()
            if not data or 'panels' not in data:
                return jsonify({'error': 'No panels provided'}), 400

            panels = data['panels']
            layout = data.get('layout', 'stacked')

            psd_bytes = panels_to_psd_bytes(panels, layout=layout)

            return send_file(
                io.BytesIO(psd_bytes),
                mimetype='application/octet-stream',
                as_attachment=True,
                download_name='manga_panels.psd'
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500


# =============================================================================
# TRAINING DATA COLLECTION ENDPOINTS
# =============================================================================

TRAINING_DATA_FOLDER = 'training_data'
os.makedirs(TRAINING_DATA_FOLDER, exist_ok=True)
os.makedirs(os.path.join(TRAINING_DATA_FOLDER, 'obb', 'images'), exist_ok=True)
os.makedirs(os.path.join(TRAINING_DATA_FOLDER, 'obb', 'labels'), exist_ok=True)


@app.route('/api/save_training_sample', methods=['POST'])
def save_training_sample():
    """
    Save a training sample with the original image and corrected bounding boxes.
    Uses OBB (Oriented Bounding Box) format for training.

    Expects:
        - multipart/form-data with 'image' file (original manga page)
        - 'boxes' field containing JSON array of panel data
        - Each box should have EITHER:
          - {x, y, width, height} for axis-aligned boxes
          - {corners: [{x, y}, ...]} for polygons (will save as OBB with 4 corners)

    Saves:
        - Image to training_data/obb/images/
        - OBB labels to training_data/obb/labels/

    OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (4 corners, normalized 0-1)
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        if 'boxes' not in request.form:
            return jsonify({'error': 'No bounding boxes provided'}), 400

        # Read the original image
        image_file = request.files['image']
        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        img_height, img_width = image.shape[:2]

        # Parse bounding boxes
        import json
        boxes = json.loads(request.form['boxes'])

        # Generate unique filename
        sample_id = str(uuid.uuid4())[:8]
        image_filename = f"manga_{sample_id}.png"
        label_filename = f"manga_{sample_id}.txt"

        # Create OBB directories if they don't exist
        obb_images_dir = os.path.join(TRAINING_DATA_FOLDER, 'obb', 'images')
        obb_labels_dir = os.path.join(TRAINING_DATA_FOLDER, 'obb', 'labels')
        os.makedirs(obb_images_dir, exist_ok=True)
        os.makedirs(obb_labels_dir, exist_ok=True)

        # Save the image to OBB directory
        obb_image_path = os.path.join(obb_images_dir, image_filename)
        cv2.imwrite(obb_image_path, image)

        # Convert boxes to OBB format
        obb_labels = []
        class_id = 0  # Single class: "panel"
        has_diagonal = False  # Track if any box is non-axis-aligned

        for box in boxes:
            # Check if box has corners (polygon) or just x,y,width,height
            if 'corners' in box and len(box['corners']) >= 3:
                corners = box['corners']

                # Check if this is a diagonal/rotated box
                # A box is diagonal if corners don't form an axis-aligned rectangle
                if len(corners) >= 4:
                    # Check if any edge is not horizontal/vertical
                    for i in range(len(corners)):
                        c1 = corners[i]
                        c2 = corners[(i + 1) % len(corners)]
                        # If neither horizontal nor vertical, it's diagonal
                        if abs(c1['x'] - c2['x']) > 2 and abs(c1['y'] - c2['y']) > 2:
                            has_diagonal = True
                            break

                # For OBB: use corners directly (take first 4 if more)
                # OBB format expects exactly 4 corners
                obb_corners = corners[:4] if len(corners) >= 4 else corners

                # Pad with duplicates if less than 4 corners
                while len(obb_corners) < 4:
                    obb_corners.append(obb_corners[-1])

                # Normalize corners for OBB
                obb_parts = [str(class_id)]
                for corner in obb_corners:
                    nx = max(0, min(1, corner['x'] / img_width))
                    ny = max(0, min(1, corner['y'] / img_height))
                    obb_parts.append(f"{nx:.6f}")
                    obb_parts.append(f"{ny:.6f}")
                obb_labels.append(' '.join(obb_parts))
            else:
                # Standard bbox format
                x = float(box['x'])
                y = float(box['y'])
                w = float(box['width'])
                h = float(box['height'])

                # For OBB: create 4 corners from bbox (axis-aligned)
                obb_corners = [
                    {'x': x, 'y': y},           # top-left
                    {'x': x + w, 'y': y},       # top-right
                    {'x': x + w, 'y': y + h},   # bottom-right
                    {'x': x, 'y': y + h}        # bottom-left
                ]
                obb_parts = [str(class_id)]
                for corner in obb_corners:
                    nx = max(0, min(1, corner['x'] / img_width))
                    ny = max(0, min(1, corner['y'] / img_height))
                    obb_parts.append(f"{nx:.6f}")
                    obb_parts.append(f"{ny:.6f}")
                obb_labels.append(' '.join(obb_parts))

        # Save OBB labels file
        obb_label_path = os.path.join(obb_labels_dir, label_filename)
        with open(obb_label_path, 'w') as f:
            f.write('\n'.join(obb_labels))

        # Count total samples
        total_obb_images = len([f for f in os.listdir(obb_images_dir) if f.endswith('.png')])

        print(f"[Training] Saved sample {sample_id}: {len(boxes)} panels, diagonal={has_diagonal}, image={image_filename}")

        return jsonify({
            'success': True,
            'sample_id': sample_id,
            'image_file': image_filename,
            'label_file': label_filename,
            'panel_count': len(boxes),
            'total_samples': total_obb_images,
            'has_diagonal': has_diagonal
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/training_stats', methods=['GET'])
def get_training_stats():
    """
    Get statistics about collected OBB training data.
    """
    try:
        obb_images_dir = os.path.join(TRAINING_DATA_FOLDER, 'obb', 'images')
        obb_labels_dir = os.path.join(TRAINING_DATA_FOLDER, 'obb', 'labels')

        # OBB stats
        obb_image_files = []
        obb_label_files = []
        if os.path.exists(obb_images_dir):
            obb_image_files = [f for f in os.listdir(obb_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if os.path.exists(obb_labels_dir):
            obb_label_files = [f for f in os.listdir(obb_labels_dir) if f.endswith('.txt')]

        # Count OBB panels
        total_obb_panels = 0
        for label_file in obb_label_files:
            with open(os.path.join(obb_labels_dir, label_file), 'r') as f:
                total_obb_panels += len([line for line in f.readlines() if line.strip()])

        # Check for existing trained model
        models_dir = 'models'
        model_exists = os.path.exists(os.path.join(models_dir, 'manga_panels_obb.pt'))

        return jsonify({
            'success': True,
            'total_images': len(obb_image_files),
            'total_labels': len(obb_label_files),
            'total_panels': total_obb_panels,
            'model_exists': model_exists,
            'model_path': 'models/manga_panels_obb.pt' if model_exists else None,
            'training_data_path': os.path.abspath(os.path.join(TRAINING_DATA_FOLDER, 'obb'))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/finetune', methods=['POST'])
def finetune_model():
    """
    Fine-tune a YOLO model on collected training data.

    Query params:
        - epochs: Number of training epochs (default: 50)
        - base_model: Base YOLO model to fine-tune (default: yolov8n.pt)
        - imgsz: Image size for training (default: 640)

    Returns:
        Path to the trained model and training metrics
    """
    if not YOLO_AVAILABLE:
        return jsonify({'error': 'YOLO not available. Install with: pip install ultralytics'}), 400

    try:
        from ultralytics import YOLO
        import yaml

        epochs = request.args.get('epochs', 50, type=int)
        base_model = request.args.get('base_model', 'yolov8n.pt')
        imgsz = request.args.get('imgsz', 640, type=int)

        images_dir = os.path.join(TRAINING_DATA_FOLDER, 'images')
        labels_dir = os.path.join(TRAINING_DATA_FOLDER, 'labels')

        # Check if we have enough training data
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) < 5:
            return jsonify({
                'error': f'Need at least 5 training samples, currently have {len(image_files)}. Save more samples first.'
            }), 400

        # Create YOLO dataset configuration
        dataset_yaml = os.path.join(TRAINING_DATA_FOLDER, 'dataset.yaml')

        # Split data into train/val (80/20)
        import random
        random.shuffle(image_files)
        split_idx = int(len(image_files) * 0.8)
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:] if split_idx < len(image_files) else image_files[-1:]

        # Create train/val directories
        train_images_dir = os.path.join(TRAINING_DATA_FOLDER, 'train', 'images')
        train_labels_dir = os.path.join(TRAINING_DATA_FOLDER, 'train', 'labels')
        val_images_dir = os.path.join(TRAINING_DATA_FOLDER, 'val', 'images')
        val_labels_dir = os.path.join(TRAINING_DATA_FOLDER, 'val', 'labels')

        for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            os.makedirs(d, exist_ok=True)

        # Copy/link files to train/val directories
        import shutil
        for img in train_images:
            src_img = os.path.join(images_dir, img)
            dst_img = os.path.join(train_images_dir, img)
            label_name = os.path.splitext(img)[0] + '.txt'
            src_label = os.path.join(labels_dir, label_name)
            dst_label = os.path.join(train_labels_dir, label_name)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

        for img in val_images:
            src_img = os.path.join(images_dir, img)
            dst_img = os.path.join(val_images_dir, img)
            label_name = os.path.splitext(img)[0] + '.txt'
            src_label = os.path.join(labels_dir, label_name)
            dst_label = os.path.join(val_labels_dir, label_name)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

        # Write dataset.yaml
        dataset_config = {
            'path': os.path.abspath(TRAINING_DATA_FOLDER),
            'train': 'train/images',
            'val': 'val/images',
            'names': {
                0: 'panel'
            }
        }

        with open(dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"[Training] Starting fine-tuning with {len(train_images)} train, {len(val_images)} val images")
        print(f"[Training] Base model: {base_model}, Epochs: {epochs}, Image size: {imgsz}")

        # Load base model and train
        model = YOLO(base_model)
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            project=os.path.join(TRAINING_DATA_FOLDER, 'runs'),
            name='manga_panels',
            exist_ok=True,
            verbose=True
        )

        # Get the best model path
        best_model_path = os.path.join(TRAINING_DATA_FOLDER, 'runs', 'manga_panels', 'weights', 'best.pt')

        # Copy to models directory for easy access
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        final_model_path = os.path.join(models_dir, 'manga_panels_finetuned.pt')
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, final_model_path)

        return jsonify({
            'success': True,
            'model_path': final_model_path,
            'train_samples': len(train_images),
            'val_samples': len(val_images),
            'epochs': epochs,
            'message': f'Model fine-tuned successfully! Use {final_model_path} for detection.'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/finetune_obb', methods=['POST'])
def finetune_obb_model():
    """
    Fine-tune a YOLOv8-OBB model on collected OBB training data.
    This model can detect rotated/diagonal bounding boxes.

    Query params:
        - epochs: Number of training epochs (default: 50)
        - base_model: Base OBB model to fine-tune (default: yolov8n-obb.pt)
        - imgsz: Image size for training (default: 640)

    Returns:
        Path to the trained OBB model and training metrics
    """
    if not YOLO_AVAILABLE:
        return jsonify({'error': 'YOLO not available. Install with: pip install ultralytics'}), 400

    try:
        from ultralytics import YOLO
        import yaml
        import shutil

        epochs = request.args.get('epochs', 50, type=int)
        base_model = request.args.get('base_model', 'yolov8n-obb.pt')  # OBB base model
        imgsz = request.args.get('imgsz', 640, type=int)

        obb_images_dir = os.path.join(TRAINING_DATA_FOLDER, 'obb', 'images')
        obb_labels_dir = os.path.join(TRAINING_DATA_FOLDER, 'obb', 'labels')

        # Check if directories exist
        if not os.path.exists(obb_images_dir) or not os.path.exists(obb_labels_dir):
            return jsonify({
                'error': 'No OBB training data found. Save some training samples with diagonal boxes first.'
            }), 400

        # Check if we have enough training data
        image_files = [f for f in os.listdir(obb_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) < 5:
            return jsonify({
                'error': f'Need at least 5 training samples, currently have {len(image_files)}. Save more samples first.'
            }), 400

        # Create OBB dataset configuration
        obb_data_folder = os.path.join(TRAINING_DATA_FOLDER, 'obb')
        dataset_yaml = os.path.join(obb_data_folder, 'dataset.yaml')

        # Split data into train/val (80/20)
        import random
        random.shuffle(image_files)
        split_idx = int(len(image_files) * 0.8)
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:] if split_idx < len(image_files) else image_files[-1:]

        # Create train/val directories for OBB
        train_images_dir = os.path.join(obb_data_folder, 'train', 'images')
        train_labels_dir = os.path.join(obb_data_folder, 'train', 'labels')
        val_images_dir = os.path.join(obb_data_folder, 'val', 'images')
        val_labels_dir = os.path.join(obb_data_folder, 'val', 'labels')

        for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            os.makedirs(d, exist_ok=True)

        # Copy files to train/val directories
        for img in train_images:
            src_img = os.path.join(obb_images_dir, img)
            dst_img = os.path.join(train_images_dir, img)
            label_name = os.path.splitext(img)[0] + '.txt'
            src_label = os.path.join(obb_labels_dir, label_name)
            dst_label = os.path.join(train_labels_dir, label_name)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

        for img in val_images:
            src_img = os.path.join(obb_images_dir, img)
            dst_img = os.path.join(val_images_dir, img)
            label_name = os.path.splitext(img)[0] + '.txt'
            src_label = os.path.join(obb_labels_dir, label_name)
            dst_label = os.path.join(val_labels_dir, label_name)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

        # Write dataset.yaml for OBB
        dataset_config = {
            'path': os.path.abspath(obb_data_folder),
            'train': 'train/images',
            'val': 'val/images',
            'names': {
                0: 'panel'
            }
        }

        with open(dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"[OBB Training] Starting fine-tuning with {len(train_images)} train, {len(val_images)} val images")
        print(f"[OBB Training] Base model: {base_model}, Epochs: {epochs}, Image size: {imgsz}")

        # Load OBB base model and train
        model = YOLO(base_model)
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            project=os.path.join(obb_data_folder, 'runs'),
            name='manga_panels_obb',
            exist_ok=True,
            verbose=True
        )

        # Get the best model path
        best_model_path = os.path.join(obb_data_folder, 'runs', 'manga_panels_obb', 'weights', 'best.pt')

        # Copy to models directory for easy access
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        final_model_path = os.path.join(models_dir, 'manga_panels_obb.pt')
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, final_model_path)

        return jsonify({
            'success': True,
            'model_path': final_model_path,
            'model_type': 'obb',
            'train_samples': len(train_images),
            'val_samples': len(val_images),
            'epochs': epochs,
            'message': f'OBB model trained successfully! Use {final_model_path} for diagonal panel detection.'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/bootstrap_obb', methods=['POST'])
def bootstrap_obb_from_yolo():
    """
    Bootstrap OBB training data by running the existing YOLO model on images
    and saving the detections as OBB format (with angle=0).

    This allows you to leverage your current YOLO model's knowledge to
    kickstart OBB training, then refine with diagonal corrections.

    Expects:
        - multipart/form-data with 'images' (multiple files) or directory path
        - Or JSON with 'image_paths' array

    Query params:
        - yolo_model: Path to YOLO model (default: models/manga109_yolo.pt)
        - confidence: Detection confidence threshold (default: 0.5)
        - max_images: Maximum images to process (default: 100)

    Returns:
        Number of samples generated
    """
    if not YOLO_AVAILABLE:
        return jsonify({'error': 'YOLO not available. Install with: pip install ultralytics'}), 400

    try:
        from yolo_detector import YOLODetector

        yolo_model = request.args.get('yolo_model', 'models/manga109_yolo.pt')
        confidence = request.args.get('confidence', 0.5, type=float)
        max_images = request.args.get('max_images', 100, type=int)

        # Check if model exists
        if not os.path.exists(yolo_model):
            return jsonify({'error': f'YOLO model not found: {yolo_model}'}), 400

        # Initialize detector
        detector = YOLODetector(model_path=yolo_model, confidence=confidence)

        # Create OBB directories
        obb_images_dir = os.path.join(TRAINING_DATA_FOLDER, 'obb', 'images')
        obb_labels_dir = os.path.join(TRAINING_DATA_FOLDER, 'obb', 'labels')
        os.makedirs(obb_images_dir, exist_ok=True)
        os.makedirs(obb_labels_dir, exist_ok=True)

        samples_created = 0
        total_panels = 0

        # Process uploaded images
        if 'images' in request.files:
            files = request.files.getlist('images')
            for file in files[:max_images]:
                if not file.filename:
                    continue

                # Read image
                file_bytes = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    continue

                # Detect panels
                panels = detector.detect_panels_only(image)
                if not panels:
                    continue

                img_height, img_width = image.shape[:2]

                # Generate unique filename
                sample_id = str(uuid.uuid4())[:8]
                image_filename = f"bootstrap_{sample_id}.png"
                label_filename = f"bootstrap_{sample_id}.txt"

                # Save image
                cv2.imwrite(os.path.join(obb_images_dir, image_filename), image)

                # Convert YOLO detections to OBB format (axis-aligned, angle=0)
                obb_labels = []
                for panel in panels:
                    x1, y1, x2, y2 = panel.bbox
                    # Create 4 corners from bbox
                    corners = [
                        (x1, y1), (x2, y1), (x2, y2), (x1, y2)
                    ]
                    # Normalize
                    obb_parts = ['0']  # class_id
                    for cx, cy in corners:
                        nx = max(0, min(1, cx / img_width))
                        ny = max(0, min(1, cy / img_height))
                        obb_parts.append(f"{nx:.6f}")
                        obb_parts.append(f"{ny:.6f}")
                    obb_labels.append(' '.join(obb_parts))

                # Save labels
                with open(os.path.join(obb_labels_dir, label_filename), 'w') as f:
                    f.write('\n'.join(obb_labels))

                samples_created += 1
                total_panels += len(panels)

                print(f"[Bootstrap] Created sample {sample_id}: {len(panels)} panels from {file.filename}")

        # Also check for existing training images to bootstrap from
        elif request.is_json:
            data = request.get_json()
            if 'use_existing' in data and data['use_existing']:
                # Use images from training_data/images
                existing_images_dir = os.path.join(TRAINING_DATA_FOLDER, 'images')
                if os.path.exists(existing_images_dir):
                    image_files = [f for f in os.listdir(existing_images_dir)
                                   if f.endswith(('.png', '.jpg', '.jpeg'))][:max_images]

                    for img_file in image_files:
                        img_path = os.path.join(existing_images_dir, img_file)
                        image = cv2.imread(img_path)
                        if image is None:
                            continue

                        panels = detector.detect_panels_only(image)
                        if not panels:
                            continue

                        img_height, img_width = image.shape[:2]

                        # Use same filename for OBB version
                        base_name = os.path.splitext(img_file)[0]
                        obb_image_path = os.path.join(obb_images_dir, img_file)
                        obb_label_path = os.path.join(obb_labels_dir, base_name + '.txt')

                        # Skip if already exists
                        if os.path.exists(obb_label_path):
                            continue

                        # Copy image
                        cv2.imwrite(obb_image_path, image)

                        # Convert to OBB format
                        obb_labels = []
                        for panel in panels:
                            x1, y1, x2, y2 = panel.bbox
                            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                            obb_parts = ['0']
                            for cx, cy in corners:
                                nx = max(0, min(1, cx / img_width))
                                ny = max(0, min(1, cy / img_height))
                                obb_parts.append(f"{nx:.6f}")
                                obb_parts.append(f"{ny:.6f}")
                            obb_labels.append(' '.join(obb_parts))

                        with open(obb_label_path, 'w') as f:
                            f.write('\n'.join(obb_labels))

                        samples_created += 1
                        total_panels += len(panels)

        return jsonify({
            'success': True,
            'samples_created': samples_created,
            'total_panels': total_panels,
            'message': f'Bootstrapped {samples_created} OBB training samples with {total_panels} panels from existing YOLO detections.'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear_training_data', methods=['POST'])
def clear_training_data():
    """
    Clear all collected OBB training data.
    """
    try:
        import shutil

        obb_dir = os.path.join(TRAINING_DATA_FOLDER, 'obb')

        # Remove and recreate OBB directories
        if os.path.exists(obb_dir):
            shutil.rmtree(obb_dir)

        os.makedirs(os.path.join(obb_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(obb_dir, 'labels'), exist_ok=True)

        return jsonify({
            'success': True,
            'message': 'Training data cleared'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Check if running in production
    is_production = os.environ.get('FLASK_ENV') == 'production' or os.environ.get('RENDER')

    print("=" * 60)
    print("Manga Panel Slicer - Enhanced Edition")
    print("=" * 60)
    print(f"Mode: {'PRODUCTION' if is_production else 'Development'}")
    print(f"YOLO Detection: {'Available' if YOLO_AVAILABLE else 'Not installed (pip install ultralytics)'}")
    print(f"Rate Limiting: {'Available' if LIMITER_AVAILABLE else 'Not installed (pip install flask-limiter)'}")
    print(f"Vetting Workflow: {'Available' if VETTING_AVAILABLE else 'Not available'}")
    print(f"PSD Export: {'Available' if PSD_AVAILABLE else 'Not available'}")
    print(f"Training Data: {os.path.abspath(TRAINING_DATA_FOLDER)}")
    print("=" * 60)

    if is_production:
        # Production: use gunicorn (this block won't run, gunicorn imports app directly)
        print("Running in production mode")
    else:
        # Development
        print("Open http://localhost:5000 in your browser")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000)
