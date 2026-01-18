"""
Vetting and Sorting Workflow for Manga Panel Slicer

Provides a workflow system for reviewing, sorting, and managing
detected panels before final export.
"""

import os
import json
import shutil
import subprocess
import platform
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import cv2
import numpy as np


class PanelStatus(Enum):
    """Status of a panel in the vetting workflow"""
    PENDING = "pending"           # Not yet reviewed
    APPROVED = "approved"         # Approved for export
    REJECTED = "rejected"         # Will not be exported
    NEEDS_EDIT = "needs_edit"     # Requires mask editing
    NEEDS_PHOTOSHOP = "needs_photoshop"  # Needs external editing
    IN_PROGRESS = "in_progress"   # Currently being edited


@dataclass
class VettedPanel:
    """A panel with vetting information"""
    index: int
    original_index: int  # Index before any sorting/filtering
    image_data: Optional[str] = None  # Base64 encoded
    image_path: Optional[str] = None  # Path to saved file
    status: PanelStatus = PanelStatus.PENDING
    confidence: float = 1.0  # Detection confidence (1.0 for CV detection)
    width: int = 0
    height: int = 0
    area: float = 0.0
    notes: str = ""
    edited: bool = False
    edit_history: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'VettedPanel':
        """Create from dictionary"""
        data = data.copy()
        data['status'] = PanelStatus(data['status'])
        return cls(**data)


@dataclass
class VettingSession:
    """A vetting session containing panels to review"""
    session_id: str
    source_image: str
    panels: List[VettedPanel]
    detection_method: str = "cv"  # "cv", "yolo", "hybrid"
    detection_params: Dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed: bool = False
    export_path: Optional[str] = None

    @property
    def pending_count(self) -> int:
        return sum(1 for p in self.panels if p.status == PanelStatus.PENDING)

    @property
    def approved_count(self) -> int:
        return sum(1 for p in self.panels if p.status == PanelStatus.APPROVED)

    @property
    def rejected_count(self) -> int:
        return sum(1 for p in self.panels if p.status == PanelStatus.REJECTED)

    @property
    def needs_edit_count(self) -> int:
        return sum(1 for p in self.panels if p.status in [
            PanelStatus.NEEDS_EDIT, PanelStatus.NEEDS_PHOTOSHOP
        ])

    @property
    def progress_percent(self) -> float:
        if not self.panels:
            return 100.0
        reviewed = sum(1 for p in self.panels if p.status != PanelStatus.PENDING)
        return (reviewed / len(self.panels)) * 100

    def get_panels_by_status(self, status: PanelStatus) -> List[VettedPanel]:
        return [p for p in self.panels if p.status == status]

    def get_approved_panels(self) -> List[VettedPanel]:
        return self.get_panels_by_status(PanelStatus.APPROVED)

    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'source_image': self.source_image,
            'panels': [p.to_dict() for p in self.panels],
            'detection_method': self.detection_method,
            'detection_params': self.detection_params,
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'completed': self.completed,
            'export_path': self.export_path,
            'stats': {
                'total': len(self.panels),
                'pending': self.pending_count,
                'approved': self.approved_count,
                'rejected': self.rejected_count,
                'needs_edit': self.needs_edit_count,
                'progress': self.progress_percent
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'VettingSession':
        panels = [VettedPanel.from_dict(p) for p in data['panels']]
        return cls(
            session_id=data['session_id'],
            source_image=data['source_image'],
            panels=panels,
            detection_method=data.get('detection_method', 'cv'),
            detection_params=data.get('detection_params', {}),
            created_at=data.get('created_at', datetime.now().isoformat()),
            modified_at=data.get('modified_at', datetime.now().isoformat()),
            completed=data.get('completed', False),
            export_path=data.get('export_path')
        )


class VettingWorkflow:
    """
    Manages the vetting workflow for manga panels.

    Features:
    - Session management (create, save, load)
    - Panel status tracking
    - Sorting and filtering
    - External editor integration (Photoshop)
    - Export functionality
    """

    def __init__(self, sessions_dir: str = 'vetting_sessions'):
        """
        Initialize the vetting workflow.

        Args:
            sessions_dir: Directory to store session files
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.current_session: Optional[VettingSession] = None
        self.photoshop_path = self._find_photoshop()

    def _find_photoshop(self) -> Optional[str]:
        """Find Photoshop installation path"""
        if platform.system() == 'Windows':
            # Common Windows paths
            paths = [
                r"C:\Program Files\Adobe\Adobe Photoshop 2024\Photoshop.exe",
                r"C:\Program Files\Adobe\Adobe Photoshop 2023\Photoshop.exe",
                r"C:\Program Files\Adobe\Adobe Photoshop CC 2019\Photoshop.exe",
                r"C:\Program Files\Adobe\Adobe Photoshop CS6 (64 Bit)\Photoshop.exe",
            ]
            for path in paths:
                if os.path.exists(path):
                    return path
        elif platform.system() == 'Darwin':  # macOS
            paths = [
                "/Applications/Adobe Photoshop 2024/Adobe Photoshop 2024.app",
                "/Applications/Adobe Photoshop 2023/Adobe Photoshop 2023.app",
            ]
            for path in paths:
                if os.path.exists(path):
                    return path
        return None

    def set_photoshop_path(self, path: str):
        """Set custom Photoshop path"""
        self.photoshop_path = path

    def create_session(
        self,
        source_image: str,
        panels_data: List[Dict],
        detection_method: str = "cv",
        detection_params: Dict = None
    ) -> VettingSession:
        """
        Create a new vetting session.

        Args:
            source_image: Path or identifier of source manga page
            panels_data: List of panel dictionaries with image data
            detection_method: Method used for detection
            detection_params: Parameters used for detection

        Returns:
            New VettingSession
        """
        import uuid
        session_id = str(uuid.uuid4())[:8]

        panels = []
        for i, pdata in enumerate(panels_data):
            panel = VettedPanel(
                index=i,
                original_index=pdata.get('index', i),
                image_data=pdata.get('image'),
                width=pdata.get('width', 0),
                height=pdata.get('height', 0),
                area=pdata.get('area', 0),
                confidence=pdata.get('confidence', 1.0)
            )
            panels.append(panel)

        self.current_session = VettingSession(
            session_id=session_id,
            source_image=source_image,
            panels=panels,
            detection_method=detection_method,
            detection_params=detection_params or {}
        )

        return self.current_session

    def save_session(self, session: Optional[VettingSession] = None) -> str:
        """
        Save session to disk.

        Args:
            session: Session to save (uses current if None)

        Returns:
            Path to saved session file
        """
        session = session or self.current_session
        if not session:
            raise ValueError("No session to save")

        session.modified_at = datetime.now().isoformat()
        session_path = self.sessions_dir / f"{session.session_id}.json"

        with open(session_path, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)

        return str(session_path)

    def load_session(self, session_id: str) -> VettingSession:
        """
        Load a session from disk.

        Args:
            session_id: Session ID to load

        Returns:
            Loaded VettingSession
        """
        session_path = self.sessions_dir / f"{session_id}.json"
        if not session_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        with open(session_path, 'r') as f:
            data = json.load(f)

        self.current_session = VettingSession.from_dict(data)
        return self.current_session

    def list_sessions(self) -> List[Dict]:
        """List all saved sessions"""
        sessions = []
        for session_file in self.sessions_dir.glob('*.json'):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                sessions.append({
                    'session_id': data['session_id'],
                    'source_image': data['source_image'],
                    'panel_count': len(data['panels']),
                    'created_at': data['created_at'],
                    'completed': data.get('completed', False),
                    'progress': data.get('stats', {}).get('progress', 0)
                })
            except Exception:
                pass
        return sorted(sessions, key=lambda x: x['created_at'], reverse=True)

    def delete_session(self, session_id: str):
        """Delete a session"""
        session_path = self.sessions_dir / f"{session_id}.json"
        if session_path.exists():
            os.remove(session_path)

    # Panel status management
    def set_panel_status(
        self,
        panel_index: int,
        status: PanelStatus,
        notes: str = ""
    ):
        """Set status for a panel"""
        if not self.current_session:
            raise ValueError("No active session")

        panel = self.current_session.panels[panel_index]
        panel.status = status
        panel.modified_at = datetime.now().isoformat()
        if notes:
            panel.notes = notes

    def approve_panel(self, panel_index: int, notes: str = ""):
        """Mark panel as approved"""
        self.set_panel_status(panel_index, PanelStatus.APPROVED, notes)

    def reject_panel(self, panel_index: int, notes: str = ""):
        """Mark panel as rejected"""
        self.set_panel_status(panel_index, PanelStatus.REJECTED, notes)

    def mark_for_edit(self, panel_index: int, notes: str = ""):
        """Mark panel as needing edit"""
        self.set_panel_status(panel_index, PanelStatus.NEEDS_EDIT, notes)

    def mark_for_photoshop(self, panel_index: int, notes: str = ""):
        """Mark panel for Photoshop editing"""
        self.set_panel_status(panel_index, PanelStatus.NEEDS_PHOTOSHOP, notes)

    def approve_all_pending(self):
        """Approve all pending panels"""
        if not self.current_session:
            return
        for panel in self.current_session.panels:
            if panel.status == PanelStatus.PENDING:
                panel.status = PanelStatus.APPROVED
                panel.modified_at = datetime.now().isoformat()

    def reject_all_pending(self):
        """Reject all pending panels"""
        if not self.current_session:
            return
        for panel in self.current_session.panels:
            if panel.status == PanelStatus.PENDING:
                panel.status = PanelStatus.REJECTED
                panel.modified_at = datetime.now().isoformat()

    # Sorting and filtering
    def sort_panels(
        self,
        by: str = 'index',
        reverse: bool = False
    ) -> List[VettedPanel]:
        """
        Sort panels by various criteria.

        Args:
            by: Sort key ('index', 'area', 'confidence', 'status', 'width', 'height')
            reverse: Reverse sort order

        Returns:
            Sorted panel list
        """
        if not self.current_session:
            return []

        key_funcs = {
            'index': lambda p: p.index,
            'area': lambda p: p.area,
            'confidence': lambda p: p.confidence,
            'status': lambda p: p.status.value,
            'width': lambda p: p.width,
            'height': lambda p: p.height,
        }

        key_func = key_funcs.get(by, key_funcs['index'])
        return sorted(self.current_session.panels, key=key_func, reverse=reverse)

    def filter_panels(
        self,
        status: Optional[PanelStatus] = None,
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
        min_confidence: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> List[VettedPanel]:
        """
        Filter panels by criteria.

        Args:
            status: Filter by status
            min_area: Minimum area
            max_area: Maximum area
            min_confidence: Minimum confidence
            tags: Must have all these tags

        Returns:
            Filtered panel list
        """
        if not self.current_session:
            return []

        panels = self.current_session.panels

        if status:
            panels = [p for p in panels if p.status == status]
        if min_area is not None:
            panels = [p for p in panels if p.area >= min_area]
        if max_area is not None:
            panels = [p for p in panels if p.area <= max_area]
        if min_confidence is not None:
            panels = [p for p in panels if p.confidence >= min_confidence]
        if tags:
            panels = [p for p in panels if all(t in p.tags for t in tags)]

        return panels

    # External editor integration
    def open_in_photoshop(self, panel_index: int, temp_dir: str = 'temp_edit') -> Optional[str]:
        """
        Open a panel in Photoshop for editing.

        Args:
            panel_index: Index of panel to edit
            temp_dir: Directory for temporary files

        Returns:
            Path to temporary file or None if Photoshop not available
        """
        if not self.photoshop_path:
            print("[Vetting] Photoshop not found. Set path with set_photoshop_path()")
            return None

        if not self.current_session:
            raise ValueError("No active session")

        panel = self.current_session.panels[panel_index]

        # Save panel to temp file
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(
            temp_dir,
            f"panel_{self.current_session.session_id}_{panel_index}.png"
        )

        # Decode and save image
        if panel.image_data:
            import base64
            img_data = panel.image_data
            if ',' in img_data:
                img_data = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_data)

            with open(temp_path, 'wb') as f:
                f.write(img_bytes)

        # Open in Photoshop
        try:
            if platform.system() == 'Windows':
                subprocess.Popen([self.photoshop_path, temp_path])
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', '-a', self.photoshop_path, temp_path])
            else:
                subprocess.Popen([self.photoshop_path, temp_path])

            panel.status = PanelStatus.IN_PROGRESS
            panel.edit_history.append(f"Opened in Photoshop: {datetime.now().isoformat()}")
            return temp_path

        except Exception as e:
            print(f"[Vetting] Error opening Photoshop: {e}")
            return None

    def open_in_default_editor(self, panel_index: int, temp_dir: str = 'temp_edit') -> Optional[str]:
        """
        Open a panel in the system's default image editor.

        Args:
            panel_index: Index of panel to edit
            temp_dir: Directory for temporary files

        Returns:
            Path to temporary file
        """
        if not self.current_session:
            raise ValueError("No active session")

        panel = self.current_session.panels[panel_index]

        # Save panel to temp file
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(
            temp_dir,
            f"panel_{self.current_session.session_id}_{panel_index}.png"
        )

        # Decode and save image
        if panel.image_data:
            import base64
            img_data = panel.image_data
            if ',' in img_data:
                img_data = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_data)

            with open(temp_path, 'wb') as f:
                f.write(img_bytes)

        # Open with default application
        try:
            if platform.system() == 'Windows':
                os.startfile(temp_path)
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', temp_path])
            else:
                subprocess.Popen(['xdg-open', temp_path])

            panel.status = PanelStatus.IN_PROGRESS
            panel.edit_history.append(f"Opened in editor: {datetime.now().isoformat()}")
            return temp_path

        except Exception as e:
            print(f"[Vetting] Error opening editor: {e}")
            return None

    def reload_edited_panel(self, panel_index: int, file_path: str) -> bool:
        """
        Reload a panel after external editing.

        Args:
            panel_index: Index of panel
            file_path: Path to edited file

        Returns:
            True if successful
        """
        if not self.current_session:
            return False

        if not os.path.exists(file_path):
            return False

        panel = self.current_session.panels[panel_index]

        # Read edited image
        with open(file_path, 'rb') as f:
            img_bytes = f.read()

        import base64
        panel.image_data = base64.b64encode(img_bytes).decode('utf-8')
        panel.edited = True
        panel.status = PanelStatus.APPROVED
        panel.edit_history.append(f"Reloaded from edit: {datetime.now().isoformat()}")
        panel.modified_at = datetime.now().isoformat()

        # Update dimensions
        img = cv2.imread(file_path)
        if img is not None:
            panel.height, panel.width = img.shape[:2]

        return True

    # Export functionality
    def export_approved(
        self,
        output_dir: str,
        prefix: str = "panel",
        format: str = "png"
    ) -> List[str]:
        """
        Export all approved panels.

        Args:
            output_dir: Output directory
            prefix: Filename prefix
            format: Image format (png, jpg, webp)

        Returns:
            List of exported file paths
        """
        if not self.current_session:
            return []

        os.makedirs(output_dir, exist_ok=True)
        exported = []

        approved = self.current_session.get_approved_panels()
        for i, panel in enumerate(approved):
            if panel.image_data:
                import base64
                img_data = panel.image_data
                if ',' in img_data:
                    img_data = img_data.split(',')[1]
                img_bytes = base64.b64decode(img_data)

                filename = f"{prefix}_{i + 1:03d}.{format}"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, 'wb') as f:
                    f.write(img_bytes)
                exported.append(filepath)

        self.current_session.export_path = output_dir
        self.current_session.completed = True

        return exported

    def get_session_stats(self) -> Dict:
        """Get statistics for current session"""
        if not self.current_session:
            return {}

        return {
            'total': len(self.current_session.panels),
            'pending': self.current_session.pending_count,
            'approved': self.current_session.approved_count,
            'rejected': self.current_session.rejected_count,
            'needs_edit': self.current_session.needs_edit_count,
            'progress': self.current_session.progress_percent,
            'completed': self.current_session.completed
        }


# Convenience function for quick vetting
def quick_vet(
    panels_data: List[Dict],
    source_image: str = "unknown",
    auto_approve_confidence: float = 0.9
) -> Tuple[List[Dict], List[Dict]]:
    """
    Quick vetting based on confidence scores.

    Args:
        panels_data: Panel data with confidence scores
        source_image: Source image identifier
        auto_approve_confidence: Auto-approve panels above this confidence

    Returns:
        Tuple of (approved_panels, needs_review_panels)
    """
    approved = []
    needs_review = []

    for panel in panels_data:
        conf = panel.get('confidence', 1.0)
        if conf >= auto_approve_confidence:
            approved.append(panel)
        else:
            needs_review.append(panel)

    return approved, needs_review


if __name__ == '__main__':
    # Test the vetting workflow
    print("Vetting Workflow Test")
    print("=" * 50)

    workflow = VettingWorkflow()
    print(f"Photoshop found: {workflow.photoshop_path}")

    # Create test session
    test_panels = [
        {'index': 0, 'width': 400, 'height': 600, 'area': 240000, 'confidence': 0.95},
        {'index': 1, 'width': 350, 'height': 500, 'area': 175000, 'confidence': 0.87},
        {'index': 2, 'width': 300, 'height': 450, 'area': 135000, 'confidence': 0.72},
    ]

    session = workflow.create_session("test_manga.png", test_panels)
    print(f"Created session: {session.session_id}")
    print(f"Stats: {workflow.get_session_stats()}")
