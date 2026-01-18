"""
PSD Export Module for Manga Panel Slicer

Exports panels as layered PSD files for Photoshop workflows.
Supports multiple export modes and layer organization.
"""

import os
import io
import base64
import struct
import zlib
from typing import List, Dict, Tuple, Optional, BinaryIO
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

# Try to import psd-tools for advanced PSD support
try:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
    PSD_TOOLS_AVAILABLE = True
except ImportError:
    PSD_TOOLS_AVAILABLE = False

# Try PIL/Pillow
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class LayerInfo:
    """Information about a PSD layer"""
    name: str
    image: np.ndarray  # BGRA or BGR format
    x: int = 0
    y: int = 0
    opacity: int = 255
    visible: bool = True
    blend_mode: str = 'normal'


class SimplePSDWriter:
    """
    Simple PSD file writer.

    Creates basic PSD files without external dependencies.
    Supports multiple layers with positioning.
    """

    # PSD blend mode codes
    BLEND_MODES = {
        'normal': b'norm',
        'multiply': b'mul ',
        'screen': b'scrn',
        'overlay': b'over',
        'darken': b'dark',
        'lighten': b'lite',
        'color_dodge': b'div ',
        'color_burn': b'idiv',
        'hard_light': b'hLit',
        'soft_light': b'sLit',
        'difference': b'diff',
        'exclusion': b'smud',
    }

    def __init__(self, width: int, height: int, channels: int = 4):
        """
        Initialize PSD writer.

        Args:
            width: Canvas width
            height: Canvas height
            channels: Number of channels (3=RGB, 4=RGBA)
        """
        self.width = width
        self.height = height
        self.channels = channels
        self.layers: List[LayerInfo] = []

    def add_layer(
        self,
        name: str,
        image: np.ndarray,
        x: int = 0,
        y: int = 0,
        opacity: int = 255,
        visible: bool = True,
        blend_mode: str = 'normal'
    ):
        """
        Add a layer to the PSD.

        Args:
            name: Layer name
            image: Image data (BGR or BGRA)
            x: X position on canvas
            y: Y position on canvas
            opacity: Layer opacity (0-255)
            visible: Layer visibility
            blend_mode: Blend mode name
        """
        # Ensure BGRA format
        if len(image.shape) == 2:
            # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            # BGR -> BGRA
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        self.layers.append(LayerInfo(
            name=name,
            image=image,
            x=x,
            y=y,
            opacity=opacity,
            visible=visible,
            blend_mode=blend_mode
        ))

    def _write_header(self, f: BinaryIO):
        """Write PSD file header"""
        # Signature: '8BPS'
        f.write(b'8BPS')
        # Version: 1
        f.write(struct.pack('>H', 1))
        # Reserved: 6 bytes of zeros
        f.write(b'\x00' * 6)
        # Channels: 4 (RGBA)
        f.write(struct.pack('>H', self.channels))
        # Height
        f.write(struct.pack('>I', self.height))
        # Width
        f.write(struct.pack('>I', self.width))
        # Depth: 8 bits
        f.write(struct.pack('>H', 8))
        # Color mode: 3 = RGB
        f.write(struct.pack('>H', 3))

    def _write_color_mode_data(self, f: BinaryIO):
        """Write color mode data section (empty for RGB)"""
        f.write(struct.pack('>I', 0))

    def _write_image_resources(self, f: BinaryIO):
        """Write image resources section"""
        # Minimal image resources
        resources = io.BytesIO()

        # Resolution info (required)
        resources.write(b'8BIM')  # Signature
        resources.write(struct.pack('>H', 0x03ED))  # Resource ID: Resolution
        resources.write(struct.pack('>H', 0))  # Pascal string (empty)
        res_data = struct.pack('>IHHIHH', 72 << 16, 1, 1, 72 << 16, 1, 1)
        resources.write(struct.pack('>I', len(res_data)))
        resources.write(res_data)

        data = resources.getvalue()
        f.write(struct.pack('>I', len(data)))
        f.write(data)

    def _write_layer_and_mask(self, f: BinaryIO):
        """Write layer and mask information section"""
        if not self.layers:
            f.write(struct.pack('>I', 0))
            return

        layer_data = io.BytesIO()

        # Layer info
        layer_info = io.BytesIO()

        # Layer count (negative = first alpha channel contains transparency)
        layer_info.write(struct.pack('>h', -len(self.layers)))

        # Layer records
        for layer in self.layers:
            h, w = layer.image.shape[:2]

            # Layer bounds
            top = layer.y
            left = layer.x
            bottom = top + h
            right = left + w

            layer_info.write(struct.pack('>i', top))
            layer_info.write(struct.pack('>i', left))
            layer_info.write(struct.pack('>i', bottom))
            layer_info.write(struct.pack('>i', right))

            # Number of channels
            layer_info.write(struct.pack('>H', 4))

            # Channel info (4 channels: R, G, B, A)
            for channel_id in [0, 1, 2, -1]:  # R, G, B, Alpha
                layer_info.write(struct.pack('>h', channel_id))
                # Channel data length (placeholder, will use raw data)
                channel_size = h * w + 2  # +2 for compression type
                layer_info.write(struct.pack('>I', channel_size))

            # Blend mode signature
            layer_info.write(b'8BIM')
            # Blend mode
            mode = self.BLEND_MODES.get(layer.blend_mode, b'norm')
            layer_info.write(mode)
            # Opacity
            layer_info.write(struct.pack('>B', layer.opacity))
            # Clipping
            layer_info.write(struct.pack('>B', 0))
            # Flags
            flags = 0 if layer.visible else 2
            layer_info.write(struct.pack('>B', flags))
            # Filler
            layer_info.write(b'\x00')

            # Extra data length
            extra_data = io.BytesIO()

            # Layer mask data (none)
            extra_data.write(struct.pack('>I', 0))

            # Blending ranges (none)
            extra_data.write(struct.pack('>I', 0))

            # Layer name (Pascal string, padded to 4 bytes)
            name_bytes = layer.name.encode('utf-8')[:255]
            name_len = len(name_bytes)
            extra_data.write(struct.pack('>B', name_len))
            extra_data.write(name_bytes)
            # Pad to 4-byte boundary
            padding = (4 - (name_len + 1) % 4) % 4
            extra_data.write(b'\x00' * padding)

            extra_bytes = extra_data.getvalue()
            layer_info.write(struct.pack('>I', len(extra_bytes)))
            layer_info.write(extra_bytes)

        # Channel image data
        for layer in self.layers:
            img = layer.image
            h, w = img.shape[:2]

            # Write each channel (B, G, R, A -> R, G, B, A for PSD)
            for channel_idx in [2, 1, 0, 3]:  # BGR(A) -> RGB(A)
                # Compression type: 0 = raw
                layer_info.write(struct.pack('>H', 0))
                # Channel data
                channel_data = img[:, :, channel_idx].tobytes()
                layer_info.write(channel_data)

        layer_info_data = layer_info.getvalue()

        # Write layer info length
        layer_data.write(struct.pack('>I', len(layer_info_data)))
        layer_data.write(layer_info_data)
        # Pad to even length
        if len(layer_info_data) % 2:
            layer_data.write(b'\x00')

        # Global layer mask info (none)
        layer_data.write(struct.pack('>I', 0))

        full_data = layer_data.getvalue()
        f.write(struct.pack('>I', len(full_data)))
        f.write(full_data)

    def _write_image_data(self, f: BinaryIO):
        """Write merged image data"""
        # Create merged/composite image
        composite = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        composite[:, :, 3] = 255  # Opaque white background

        # Simple compositing (back to front)
        for layer in self.layers:
            if not layer.visible:
                continue

            img = layer.image
            h, w = img.shape[:2]
            x, y = layer.x, layer.y

            # Bounds check
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(self.width, x + w), min(self.height, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            # Source region
            sx1, sy1 = x1 - x, y1 - y
            sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

            # Alpha blend
            src = img[sy1:sy2, sx1:sx2]
            dst = composite[y1:y2, x1:x2]

            alpha = src[:, :, 3:4].astype(float) / 255.0 * (layer.opacity / 255.0)
            composite[y1:y2, x1:x2] = (
                src[:, :, :3] * alpha + dst[:, :, :3] * (1 - alpha)
            ).astype(np.uint8)
            composite[y1:y2, x1:x2, 3] = 255

        # Compression: 0 = raw
        f.write(struct.pack('>H', 0))

        # Write channels (R, G, B, A)
        for channel_idx in [2, 1, 0, 3]:  # BGR(A) -> RGB(A)
            channel_data = composite[:, :, channel_idx].tobytes()
            f.write(channel_data)

    def save(self, path: str):
        """
        Save PSD file.

        Args:
            path: Output file path
        """
        with open(path, 'wb') as f:
            self._write_header(f)
            self._write_color_mode_data(f)
            self._write_image_resources(f)
            self._write_layer_and_mask(f)
            self._write_image_data(f)


def export_panels_as_psd(
    panels: List[Dict],
    output_path: str,
    canvas_width: Optional[int] = None,
    canvas_height: Optional[int] = None,
    layout: str = 'stacked',
    include_composite: bool = True
) -> str:
    """
    Export panels as a layered PSD file.

    Args:
        panels: List of panel dictionaries with 'image' (base64), 'width', 'height'
        output_path: Path for output PSD file
        canvas_width: Canvas width (auto-calculated if None)
        canvas_height: Canvas height (auto-calculated if None)
        layout: Layout mode ('stacked', 'grid', 'original')
        include_composite: Include merged background layer

    Returns:
        Path to saved PSD file
    """
    if not panels:
        raise ValueError("No panels to export")

    # Decode panel images
    decoded_panels = []
    for i, panel in enumerate(panels):
        img_data = panel.get('image', '')
        if ',' in img_data:
            img_data = img_data.split(',')[1]

        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        decoded_panels.append({
            'image': img,
            'width': img.shape[1],
            'height': img.shape[0],
            'index': panel.get('index', i)
        })

    if not decoded_panels:
        raise ValueError("No valid panel images")

    # Calculate canvas size
    if layout == 'stacked':
        # Stack vertically
        max_width = max(p['width'] for p in decoded_panels)
        total_height = sum(p['height'] for p in decoded_panels)
        canvas_width = canvas_width or max_width
        canvas_height = canvas_height or total_height
    elif layout == 'grid':
        # 2-column grid
        cols = 2
        rows = (len(decoded_panels) + cols - 1) // cols
        max_width = max(p['width'] for p in decoded_panels)
        max_height = max(p['height'] for p in decoded_panels)
        canvas_width = canvas_width or (max_width * cols)
        canvas_height = canvas_height or (max_height * rows)
    else:  # 'original' - place at origin
        max_width = max(p['width'] for p in decoded_panels)
        max_height = max(p['height'] for p in decoded_panels)
        canvas_width = canvas_width or max_width
        canvas_height = canvas_height or max_height

    # Create PSD
    psd = SimplePSDWriter(canvas_width, canvas_height, channels=4)

    # Add panels as layers
    y_offset = 0
    for i, panel in enumerate(decoded_panels):
        img = panel['image']
        idx = panel['index']

        if layout == 'stacked':
            x, y = 0, y_offset
            y_offset += panel['height']
        elif layout == 'grid':
            col = i % 2
            row = i // 2
            max_w = max(p['width'] for p in decoded_panels)
            max_h = max(p['height'] for p in decoded_panels)
            x = col * max_w
            y = row * max_h
        else:
            x, y = 0, 0

        psd.add_layer(
            name=f"Panel {idx + 1}",
            image=img,
            x=x,
            y=y,
            opacity=255,
            visible=True
        )

    # Save
    psd.save(output_path)
    return output_path


def export_single_panel_psd(
    image: np.ndarray,
    output_path: str,
    layer_name: str = "Panel"
) -> str:
    """
    Export a single panel as PSD with transparency preserved.

    Args:
        image: Panel image (BGR or BGRA)
        output_path: Output path
        layer_name: Name for the layer

    Returns:
        Path to saved file
    """
    h, w = image.shape[:2]
    psd = SimplePSDWriter(w, h, channels=4)
    psd.add_layer(layer_name, image)
    psd.save(output_path)
    return output_path


def export_with_mask_layers(
    original_image: np.ndarray,
    panel_mask: np.ndarray,
    panels: List[Dict],
    output_path: str
) -> str:
    """
    Export PSD with original image, mask, and extracted panels as layers.

    Args:
        original_image: Original manga page
        panel_mask: Detection mask
        panels: Extracted panel data
        output_path: Output path

    Returns:
        Path to saved file
    """
    h, w = original_image.shape[:2]
    psd = SimplePSDWriter(w, h, channels=4)

    # Add original as background
    psd.add_layer("Original", original_image, opacity=255)

    # Add mask as semi-transparent overlay
    mask_colored = cv2.cvtColor(panel_mask, cv2.COLOR_GRAY2BGRA)
    mask_colored[:, :, 0] = 0  # Remove blue
    mask_colored[:, :, 1] = 0  # Remove green
    mask_colored[:, :, 2] = panel_mask  # Red channel
    mask_colored[:, :, 3] = (panel_mask * 0.5).astype(np.uint8)  # Semi-transparent

    psd.add_layer("Detection Mask", mask_colored, opacity=128, visible=False)

    # Add each panel
    for panel in panels:
        img_data = panel.get('image', '')
        if ',' in img_data:
            img_data = img_data.split(',')[1]

        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img is not None:
            idx = panel.get('index', 0)
            psd.add_layer(f"Panel {idx + 1}", img, visible=False)

    psd.save(output_path)
    return output_path


def panels_to_psd_bytes(panels: List[Dict], layout: str = 'stacked') -> bytes:
    """
    Export panels to PSD and return as bytes (for web download).

    Args:
        panels: Panel data
        layout: Layout mode

    Returns:
        PSD file as bytes
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.psd', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_panels_as_psd(panels, tmp_path, layout=layout)
        with open(tmp_path, 'rb') as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def is_psd_export_available() -> Dict[str, bool]:
    """Check PSD export capabilities"""
    return {
        'simple_psd': True,  # Built-in simple writer always available
        'psd_tools': PSD_TOOLS_AVAILABLE,
        'pillow': PIL_AVAILABLE
    }


if __name__ == '__main__':
    print("PSD Export Module Test")
    print("=" * 50)
    print(f"Capabilities: {is_psd_export_available()}")

    # Create test image
    test_img = np.zeros((100, 200, 3), dtype=np.uint8)
    test_img[:, :, 2] = 255  # Red

    # Test single panel export
    print("\nTesting single panel export...")
    export_single_panel_psd(test_img, "test_panel.psd")
    print("Created: test_panel.psd")

    # Clean up
    if os.path.exists("test_panel.psd"):
        os.remove("test_panel.psd")
        print("Cleaned up test file")
