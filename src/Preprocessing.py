# -*- coding: utf-8 -*-
"""
DICStudio - Python-based 2D Digital Image Correlation (DIC) software.

This file is part of DICStudio, a full Python port of the Ncorr 2D DIC MATLAB
software with additional tools and GUI features.

Copyright (c) 2025, Seyed Ali Mirjafari
All rights reserved.

Portions of this work are based on:
Blaber, J., Adair, B., & Antoniou, A. (2015).
"Ncorr: Open-Source 2D Digital Image Correlation Matlab Software",
Experimental Mechanics, 55(6), 1105–1122.

DICStudio is distributed under the BSD 3-Clause License.
See the LICENSE file in the repository root for details.

Developed for research and educational use. No warranty is provided;
use at your own risk.

"""
import sys
import os
import io
import re
from datetime import datetime
import threading
import random
import math
import json
import shutil
import tempfile
import ncorr
from pathlib import Path
from typing import Optional, List, Tuple, Dict
# Core Libraries
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageTk

# GUI Library
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox, QRadioButton,
    QSlider, QListWidget, QFileDialog, QMessageBox, QGridLayout, QFormLayout,
    QTextEdit, QSplitter, QScrollArea, QStackedWidget, QProgressBar, QDialog,
    QProgressDialog, QToolBar, QSpinBox, QColorDialog, QDialogButtonBox, QListWidgetItem,
    QSizePolicy  
)
from PySide6.QtGui import (
    QPixmap, QImage, QIcon, QPainter, QPen, QBrush, QAction, QColor, QFont
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject, QPoint, QSettings, QRect, QPointF, QBuffer, QRectF

# Processing Libraries
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi
from scipy.stats import entropy
from scipy.interpolate import griddata
from tqdm import tqdm
import imageio
import piexif

# Matplotlib for Virtual Lab
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Optional Background Removal
REMBG_AVAILABLE = False
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    pass # Keep it False

# Global variable to track if user has agreed to download rembg model
rembg_permission_granted = False

# -------------------- Utility Functions & Custom Widgets --------------------
def check_rembg_model():
    """Checks for the rembg model and asks for permission to download."""
    global rembg_permission_granted
    if rembg_permission_granted:
        return True
    if not REMBG_AVAILABLE:
        return False

    # Path where rembg stores its models
    model_path = Path(os.path.expanduser("~")) / ".u2net" / "u2net.onnx"
    if model_path.exists():
        rembg_permission_granted = True
        return True

    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Question)
    msg_box.setText("Background removal feature requires a one-time model download (~176 MB).")
    msg_box.setInformativeText("DICStudio needs to download the 'u2net.onnx' model file from the official rembg repository. Do you want to proceed?")
    msg_box.setWindowTitle("Download Required")
    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg_box.setDefaultButton(QMessageBox.Yes)

    if msg_box.exec() == QMessageBox.Yes:
        rembg_permission_granted = True
        return True
    else:
        QMessageBox.information(None, "Download Canceled", "Background removal will be disabled for this session.")
        return False

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def natural_key(string_):
    """Sorts strings with numbers in a human-friendly way."""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    """Convert a PIL Image to QPixmap."""
    try:
        mode = pil_img.mode
        if mode not in ("RGB", "RGBA", "L", "1"):
            pil_img = pil_img.convert("RGB")
            mode = "RGB"

        if mode == "RGB":
            data = pil_img.tobytes("raw", "RGB")
            qimage = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGB888).rgbSwapped()
        elif mode == "RGBA":
            data = pil_img.tobytes("raw", "RGBA")
            qimage = QImage(data, pil_img.width, pil_img.height, QImage.Format_ARGB32)
        elif mode in ("L", "1"): # Grayscale or Binary
            data = pil_img.tobytes("raw", "L")
            qimage = QImage(data, pil_img.width, pil_img.height, QImage.Format_Grayscale8)

        return QPixmap.fromImage(qimage)
    except Exception as e:
        print(f"Error converting PIL to QPixmap: {e}")
        return QPixmap()

def qpixmap_to_pil(qpixmap: QPixmap) -> Image.Image:
    """Converts a QPixmap to a PIL Image."""
    qimage = qpixmap.toImage()
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    # Save in a lossless format like PNG
    qimage.save(buffer, "PNG")
    pil_img = Image.open(io.BytesIO(buffer.data()))
    return pil_img

class FitLabel(QLabel):
    """A QLabel that automatically scales its pixmap to fit its size."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pixmap = QPixmap()
        self.setMinimumSize(1, 1)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update_pixmap()

    def pixmap(self):
        return self._pixmap

    def resizeEvent(self, event):
        self.update_pixmap()
        super().resizeEvent(event)

    def update_pixmap(self):
        if not self._pixmap or self._pixmap.isNull():
            super().setPixmap(QPixmap())
            return
        scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(scaled_pixmap)

class ClickableLabel(QLabel):
    """A QLabel that emits a 'clicked' signal with QPoint coordinates on mouse press."""
    clicked = Signal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True) # Optional: if you want mouseMoveEvent

    def mousePressEvent(self, event):
        self.clicked.emit(event.position().toPoint())
        super().mousePressEvent(event)

# =============================================================================
# START: Re-written ROITab section (Replace the old ROITab)
# =============================================================================

# This advanced ROIDrawer is imported from 'custom_widgets.py' and included here for completeness.
class ROIDrawer(QWidget):
    """
    An advanced widget for drawing, selecting, and editing ROI layers with the following features:
    - Layer management for each shape (rectangle, ellipse, polygon, brush).
    - Selection, movement, and resizing of layers with control handles.
    - Brush tool for freehand drawing with adjustable thickness.
    - Separate coloring for 'add' (green) and 'subtract' (red) modes.
    - `mask_changed` signal to send the final mask and `layers_changed` to update the layer list.
    """
    mask_changed = Signal(np.ndarray)
    layers_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self.img_np: Optional[np.ndarray] = None
        self.final_mask: Optional[np.ndarray] = None
        self.layers: List[dict] = []
        self.current_mode: Tuple[str, str] = ("rect", "add")
        self.selected_layer_idx: Optional[int] = None
        self.active_handle: Optional[str] = None
        self.is_drawing = False
        self.draw_start_pt: Optional[QPointF] = None
        self.current_shape: Optional[QRectF] = None
        self.brush_stroke: List[QPointF] = []
        self.brush_size = 20
        self.is_panning = False
        self.pan_last_pos = QPointF()
        self.offset = QPointF(0, 0)
        self.zoom = 1.0

    def set_image(self, img_np: np.ndarray, existing_mask: Optional[np.ndarray] = None):
        if img_np.dtype != np.uint8:
            img_np = (np.clip(img_np, 0, 255)).astype(np.uint8)
        self.img_np = img_np
        self.final_mask = np.zeros_like(img_np, dtype=bool)
        self.layers.clear()

        if existing_mask is not None:
            self.layers.append({
                "type": "imported",
                "mode": "add",
                "mask": existing_mask.copy().astype(bool)
            })

        self.fit_to_view()
        self._recomposite_mask()
        self.update()

    def fit_to_view(self):
        if self.img_np is None: return
        img_h, img_w = self.img_np.shape
        view_w, view_h = self.width(), self.height()
        if img_w == 0 or img_h == 0: return

        scale_w = view_w / img_w
        scale_h = view_h / img_h
        self.zoom = min(scale_w, scale_h)
        self.offset = QPointF(
            (view_w - img_w * self.zoom) / 2.0,
            (view_h - img_h * self.zoom) / 2.0
        )

    def set_mode(self, shape: str, action: str = "add"):
        self.current_mode = (shape, action)
        self.deselect_layer()
        self.update()

    def set_brush_size(self, size: int):
        self.brush_size = max(1, size)

    def clear_all(self):
        self.layers.clear()
        self.deselect_layer()
        self._recomposite_mask()

    def remove_layer(self, index: int):
        if 0 <= index < len(self.layers):
            del self.layers[index]
            self.deselect_layer()
            self._recomposite_mask()

    def deselect_layer(self):
        self.selected_layer_idx = None
        self.active_handle = None
        self.update()

    def _world_to_view(self, p: QPointF) -> QPointF:
        return (p * self.zoom) + self.offset

    def _view_to_world(self, p: QPointF) -> QPointF:
        if abs(self.zoom) < 1e-9: return QPointF()
        return (p - self.offset) / self.zoom

    def mousePressEvent(self, ev):
        pos_world = self._view_to_world(ev.position())
        shape, action = self.current_mode

        if ev.button() == Qt.MiddleButton or shape == "pan":
            self.is_panning = True
            self.pan_last_pos = ev.position()
            return

        if ev.button() != Qt.LeftButton: return

        if self.selected_layer_idx is not None:
            handles = self._get_handles_for_selected_layer()
            for handle_name, rect in handles.items():
                if rect.contains(ev.position()):
                    self.active_handle = handle_name
                    self.is_drawing = True
                    self.draw_start_pt = pos_world
                    return

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer["type"] in ("rect", "ellipse"):
                if layer["geom"].contains(pos_world):
                    self.selected_layer_idx = i
                    self.is_drawing = True
                    self.draw_start_pt = pos_world
                    self.update()
                    self.layers_changed.emit(self.get_layer_names())
                    return

        self.deselect_layer()
        if shape in ("rect", "ellipse"):
            self.is_drawing = True
            self.draw_start_pt = pos_world
            self.current_shape = QRectF(pos_world, pos_world)
        elif shape == "brush":
            self.is_drawing = True
            self.brush_stroke = [pos_world]

    def mouseMoveEvent(self, ev):
        pos_world = self._view_to_world(ev.position())

        if self.is_panning:
            delta = ev.position() - self.pan_last_pos
            self.offset += delta
            self.pan_last_pos = ev.position()
            self.update()
            return

        if not self.is_drawing:
            self.update()
            return

        if self.active_handle:
            self._resize_or_move_layer(pos_world)
        elif self.selected_layer_idx is not None and self.layers[self.selected_layer_idx]['type'] in ("rect", "ellipse"):
            delta = pos_world - self.draw_start_pt
            self.layers[self.selected_layer_idx]['geom'].translate(delta)
            self.draw_start_pt = pos_world
        elif self.current_mode[0] in ("rect", "ellipse"):
            self.current_shape = QRectF(self.draw_start_pt, pos_world).normalized()
        elif self.current_mode[0] == "brush":
            self.brush_stroke.append(pos_world)

        self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MiddleButton or self.current_mode[0] == "pan":
            self.is_panning = False
            return

        if not self.is_drawing: return
        self.is_drawing = False

        shape, action = self.current_mode

        if self.active_handle or self.selected_layer_idx is not None:
            self._recomposite_mask()
        elif self.current_shape and self.current_shape.width() > 1 and self.current_shape.height() > 1:
            self.layers.append({
                "type": shape, "mode": action, "geom": QRectF(self.current_shape)
            })
            self._recomposite_mask()
        elif shape == "brush" and len(self.brush_stroke) > 1:
            self.layers.append({
                "type": "brush", "mode": action,
                "points": list(self.brush_stroke), "size": self.brush_size
            })
            self._recomposite_mask()

        self.current_shape = None
        self.brush_stroke = []
        self.active_handle = None

    def wheelEvent(self, ev):
        if self.img_np is None: return
        factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        mouse_pos_view = ev.position()
        mouse_pos_world = self._view_to_world(mouse_pos_view)

        self.zoom *= factor
        new_offset = mouse_pos_view - mouse_pos_world * self.zoom
        self.offset = new_offset
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#333"))
        if self.img_np is None:
            p.setPen(Qt.white)
            p.drawText(self.rect(), Qt.AlignCenter, "Load an image to begin ROI definition.")
            p.end(); return

        p.save()
        p.translate(self.offset)
        p.scale(self.zoom, self.zoom)

        pix = QPixmap.fromImage(pil_to_qpixmap(Image.fromarray(self.img_np)).toImage())
        p.drawPixmap(0, 0, pix)

        if self.final_mask is not None and self.final_mask.any():
            mask_color = QColor(0, 255, 0, 70)
            mask_img = np.zeros((*self.final_mask.shape, 4), dtype=np.uint8)
            mask_img[self.final_mask] = [mask_color.red(), mask_color.green(), mask_color.blue(), mask_color.alpha()]
            q_mask_img = QImage(mask_img.data, mask_img.shape[1], mask_img.shape[0], QImage.Format_RGBA8888)
            p.drawImage(0, 0, q_mask_img)

        pen_width = 1.5 / self.zoom
        for i, layer in enumerate(self.layers):
            color = QColor("lime") if layer["mode"] == "add" else QColor("red")
            is_selected = (i == self.selected_layer_idx)
            current_pen = QPen(color, pen_width * (2.5 if is_selected else 1.2))
            p.setPen(current_pen)
            p.setBrush(Qt.NoBrush)

            if layer["type"] in ("rect", "ellipse"):
                geom = layer["geom"]
                if layer["type"] == "rect": p.drawRect(geom)
                else: p.drawEllipse(geom)
        
        if self.current_shape:
            color = QColor("lime") if self.current_mode[1] == "add" else QColor("red")
            p.setPen(QPen(color, pen_width, Qt.DashLine))
            if self.current_mode[0] == "rect": p.drawRect(self.current_shape)
            else: p.drawEllipse(self.current_shape)

        if self.brush_stroke:
            color = QColor("lime") if self.current_mode[1] == "add" else QColor("red")
            p.setPen(QPen(color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPolyline(self.brush_stroke)
        
        p.restore()

        if self.selected_layer_idx is not None and not self.is_panning:
            handles = self._get_handles_for_selected_layer()
            p.setPen(Qt.NoPen); p.setBrush(QColor("yellow"))
            for rect in handles.values(): p.drawRect(rect)
        
        if self.current_mode[0] == 'brush' and not self.is_drawing:
            cursor_pos = self.mapFromGlobal(self.cursor().pos())
            p.setPen(QPen(QColor(255, 255, 0, 200), 1.5))
            p.setBrush(Qt.NoBrush)
            radius = (self.brush_size * self.zoom) / 2
            p.drawEllipse(cursor_pos, radius, radius)

        p.end()

    def _recomposite_mask(self):
        if self.img_np is None: return
        h, w = self.img_np.shape
        self.final_mask = np.zeros((h, w), dtype=bool)

        for layer in self.layers:
            layer_mask = np.zeros((h, w), dtype=np.uint8)
            if layer["type"] in ("rect", "ellipse"):
                r = layer["geom"].toRect()
                if layer["type"] == "rect": cv2.rectangle(layer_mask, (r.x(), r.y()), (r.right(), r.bottom()), 1, -1)
                else:
                    center = (r.center().x(), r.center().y())
                    axes = (r.width() // 2, r.height() // 2)
                    cv2.ellipse(layer_mask, center, axes, 0, 0, 360, 1, -1)
            elif layer["type"] == "brush":
                pts = np.array([[p.x(), p.y()] for p in layer["points"]], np.int32)
                size = int(layer["size"])
                cv2.polylines(layer_mask, [pts], isClosed=False, color=1, thickness=size, lineType=cv2.LINE_AA)
            elif layer["type"] == "imported":
                layer_mask = layer["mask"].astype(np.uint8)

            if layer["mode"] == "add": self.final_mask |= (layer_mask > 0)
            else: self.final_mask &= ~(layer_mask > 0)

        self.update()
        self.mask_changed.emit(self.final_mask.copy())
        self.layers_changed.emit(self.get_layer_names())

    def get_layer_names(self) -> List[str]:
        return [f"Layer {i + 1}: {l['type']} ({l['mode']})" for i, l in enumerate(self.layers)]

    def _get_handles_for_selected_layer(self) -> Dict[str, QRectF]:
        if self.selected_layer_idx is None: return {}
        layer = self.layers[self.selected_layer_idx]
        if layer["type"] not in ("rect", "ellipse"): return {}
        geom = layer["geom"]; size = 8.0
        tl, tr = self._world_to_view(geom.topLeft()), self._world_to_view(geom.topRight())
        bl, br = self._world_to_view(geom.bottomLeft()), self._world_to_view(geom.bottomRight())
        return {
            "tl": QRectF(tl.x() - size/2, tl.y() - size/2, size, size), "tr": QRectF(tr.x() - size/2, tr.y() - size/2, size, size),
            "bl": QRectF(bl.x() - size/2, bl.y() - size/2, size, size), "br": QRectF(br.x() - size/2, br.y() - size/2, size, size),
        }

    def _resize_or_move_layer(self, pos_world: QPointF):
        if self.selected_layer_idx is None or self.active_handle is None: return
        layer, geom = self.layers[self.selected_layer_idx], self.layers[self.selected_layer_idx]["geom"]
        fixed_corner = QPointF()
        if self.active_handle == "tl": fixed_corner = geom.bottomRight()
        elif self.active_handle == "tr": fixed_corner = geom.bottomLeft()
        elif self.active_handle == "bl": fixed_corner = geom.topRight()
        elif self.active_handle == "br": fixed_corner = geom.topLeft()
        if not fixed_corner.isNull(): layer["geom"] = QRectF(fixed_corner, pos_world).normalized()


class ROITab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.roi_drawer = ROIDrawer()
        self.image_path = None
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Left Panel: Controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)

        # Input Group
        input_group = QGroupBox("1. Input Image")
        input_layout = QVBoxLayout(input_group)
        self.load_from_tab1_btn = QPushButton("Load from Frame Processor")
        self.load_standalone_btn = QPushButton("Load Standalone Image...")
        input_layout.addWidget(self.load_from_tab1_btn)
        input_layout.addWidget(self.load_standalone_btn)
        left_layout.addWidget(input_group)

        # Auto-Masking Group
        auto_mask_group = QGroupBox("2. Automatic Mask Generation")
        auto_mask_layout = QVBoxLayout(auto_mask_group)
        self.generate_mask_btn = QPushButton("Create Foreground Mask (rembg)")
        if not REMBG_AVAILABLE:
            self.generate_mask_btn.setText("Create Mask (rembg not installed)")
            self.generate_mask_btn.setToolTip("Install 'rembg' for this feature.")
            self.generate_mask_btn.setEnabled(False)
        auto_mask_layout.addWidget(self.generate_mask_btn)
        left_layout.addWidget(auto_mask_group)

        # ROI Management Group
        manage_group = QGroupBox("3. ROI Management")
        manage_layout = QHBoxLayout(manage_group)
        self.save_roi_btn = QPushButton("Save ROI Mask")
        self.load_roi_btn = QPushButton("Load ROI Mask")
        manage_layout.addWidget(self.save_roi_btn)
        manage_layout.addWidget(self.load_roi_btn)
        left_layout.addWidget(manage_group)

        # Layer Management
        layer_group = QGroupBox("4. Layer Management")
        layer_layout = QVBoxLayout(layer_group)
        self.layer_list = QListWidget()
        self.layer_list.itemClicked.connect(self.on_layer_selected)
        
        layer_btn_layout = QHBoxLayout()
        self.del_layer_btn = QPushButton("Delete Selected")
        self.clear_all_btn = QPushButton("Clear All")
        layer_btn_layout.addWidget(self.del_layer_btn)
        layer_btn_layout.addWidget(self.clear_all_btn)
        
        layer_layout.addWidget(self.layer_list)
        layer_layout.addLayout(layer_btn_layout)
        left_layout.addWidget(layer_group)
        left_layout.addStretch()

        # Right Panel: ROI Drawer + Toolbar
        right_group = QGroupBox("ROI Editor")
        right_layout = QVBoxLayout(right_group)

        toolbar = QToolBar("ROI Tools")
        self.tool_actions = {}
        tool_data = [
            ("rect_add", "Add Rectangle"), ("rect_sub", "Subtract Rectangle"),
            ("ellipse_add", "Add Ellipse"), ("ellipse_sub", "Subtract Ellipse"),
            ("brush_add", "Add Brush"), ("brush_sub", "Subtract Brush"),
            ("pan", "Pan/Zoom Tool")
        ]
        for key, text in tool_data:
            action = QAction(text, self, checkable=True)
            shape, mode = key.split('_') if '_' in key else (key, 'add')
            action.triggered.connect(lambda checked, s=shape, m=mode: self.select_tool(s, m))
            self.tool_actions[key] = action
            toolbar.addAction(action)
        
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Brush Size:"))
        self.brush_size_spinbox = QSpinBox()
        self.brush_size_spinbox.setRange(1, 200)
        self.brush_size_spinbox.setValue(20)
        self.brush_size_spinbox.valueChanged.connect(self.roi_drawer.set_brush_size)
        toolbar.addWidget(self.brush_size_spinbox)
        
        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.roi_drawer, 1)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_group)
        splitter.setSizes([320, 700])
        layout.addWidget(splitter)
        if hasattr(self.main_window, '_splitters') and isinstance(self.main_window._splitters, list):
            self.main_window._splitters.append(splitter)

        # Connections
        self.load_from_tab1_btn.clicked.connect(self.load_image_from_tab1)
        self.load_standalone_btn.clicked.connect(self.load_standalone_image)
        self.generate_mask_btn.clicked.connect(self.generate_auto_mask)
        self.save_roi_btn.clicked.connect(self.save_roi)
        self.load_roi_btn.clicked.connect(self.load_roi)
        self.del_layer_btn.clicked.connect(lambda: self.roi_drawer.remove_layer(self.layer_list.currentRow()))
        self.clear_all_btn.clicked.connect(self.roi_drawer.clear_all)
        self.roi_drawer.layers_changed.connect(self.update_layer_list)

        self.select_tool("rect", "add") # Set default tool

    def select_tool(self, shape, mode):
        self.roi_drawer.set_mode(shape, mode)
        key = f"{shape}_{mode}" if '_' not in shape else shape
        for action_key, action in self.tool_actions.items():
            action.setChecked(action_key == key)

    @Slot(list)
    def update_layer_list(self, names):
        self.layer_list.clear()
        self.layer_list.addItems(names)
        if self.roi_drawer.selected_layer_idx is not None and 0 <= self.roi_drawer.selected_layer_idx < self.layer_list.count():
            self.layer_list.setCurrentRow(self.roi_drawer.selected_layer_idx)

    @Slot(QListWidgetItem)
    def on_layer_selected(self, item):
        if item:
            row = self.layer_list.row(item)
            self.roi_drawer.selected_layer_idx = row
            self.roi_drawer.update()

    def set_background_image(self, image_path):
        if image_path and os.path.exists(image_path):
            self.image_path = image_path
            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None: raise Exception("Failed to load image with OpenCV")
                self.roi_drawer.set_image(img)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load or process the image: {e}")
        else:
            QMessageBox.warning(self, "Warning", "Could not find a valid reference image.")

    @Slot()
    def load_image_from_tab1(self):
        ref_image_path = self.main_window.get_reference_image_path()
        if ref_image_path:
            self.set_background_image(ref_image_path)
        else:
            QMessageBox.warning(self, "Warning", "No image processed in 'Frame Processor' tab. Please process images or load a standalone image.")

    @Slot()
    def load_standalone_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Image for ROI", "", "Image Files (*.png *.jpg *.jpeg *.tif)")
        if path:
            self.set_background_image(path)

    @Slot()
    def generate_auto_mask(self):
        if self.roi_drawer.img_np is None:
            QMessageBox.warning(self, "Warning", "Please load an image before generating a mask.")
            return
        if not check_rembg_model(): return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # The rembg library works best with file paths or bytes.
            # Reading from the stored path ensures format compatibility.
            if not self.image_path:
                raise Exception("Image path is not available. Please load the image again.")
            
            with open(self.image_path, 'rb') as f_in:
                input_bytes = f_in.read()

            session = new_session("u2net")
            output_bytes = remove(input_bytes, session=session)

            img_rgba = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
            alpha = np.array(img_rgba)[:, :, 3]
            mask = (alpha > 50) # Create boolean mask

            # Add the generated mask as a new "imported" layer
            self.roi_drawer.layers.append({
                "type": "imported",
                "mode": "add",
                "mask": mask
            })
            self.roi_drawer._recomposite_mask() # Trigger update
            QMessageBox.information(self, "Success", "Automatic mask generated and added as a new layer.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate mask: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def get_roi_mask(self):
        return self.roi_drawer.final_mask

    @Slot()
    def save_roi(self):
        mask = self.get_roi_mask()
        if mask is None or not mask.any():
            QMessageBox.warning(self, "Warning", "No ROI mask to save (mask is empty).")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save ROI Mask", "", "PNG Files (*.png)")
        if path:
            try:
                # Convert boolean mask to uint8 image (0 and 255) for saving
                mask_image = (mask * 255).astype(np.uint8)
                imageio.imwrite(path, mask_image)
                QMessageBox.information(self, "Success", f"ROI mask saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save ROI: {e}")

    @Slot()
    def load_roi(self):
        if self.roi_drawer.img_np is None:
            QMessageBox.warning(self, "Warning", "Please load a background image first before loading an ROI mask.")
            return

        path, _ = QFileDialog.getOpenFileName(self, "Load ROI Mask", "", "Image Files (*.png *.jpg *.bmp *.tif)")
        if path:
            try:
                mask_img = imageio.imread(path)
                mask = mask_img > 128 # Convert to boolean
                
                if mask.shape != self.roi_drawer.img_np.shape:
                     QMessageBox.warning(self, "Shape Mismatch", "The loaded mask dimensions do not match the background image.")
                     return

                # Add the loaded mask as a new "imported" layer
                self.roi_drawer.layers.append({
                    "type": "imported",
                    "mode": "add",
                    "mask": mask
                })
                self.roi_drawer._recomposite_mask()
                QMessageBox.information(self, "Success", f"ROI mask loaded as a new layer from {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load ROI: {e}")


# -------------------- Worker Threads --------------------
class Worker(QObject):
    finished = Signal()
    progress = Signal(int, str)  # Updated: (percentage, message)
    error = Signal(str)
    result = Signal(dict)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        try:
            res = self.func(self.progress.emit, *self.args, **self.kwargs)
            self.result.emit(res or {})
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()
# =============================================================================
# WORKER CLASS (FIXED: Sub-pixel refinement logic corrected for higher accuracy)
# =============================================================================
class CalibrationWorker(QThread):
    progress = Signal(int, str)
    corners_found = Signal(str, object)
    finished = Signal(bool, str, object, object, float)

    def __init__(self, image_paths, checkerboard_size, square_size):
        super().__init__()
        self.image_paths = image_paths
        self.rows, self.cols = checkerboard_size
        self.square_size = square_size
        self.camera_matrix = None
        self.dist_coeffs = None
        self.reprojection_error = float('inf')

    def run(self):
        try:
            self.progress.emit(0, "Initializing...")
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((self.rows * self.cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
            objp *= self.square_size

            objpoints = []
            imgpoints = []
            total_images = len(self.image_paths)
            found_count = 0
            gray_shape = None

            for i, fname in enumerate(self.image_paths):
                self.progress.emit(int(50 * (i + 1) / total_images), f"Processing {os.path.basename(fname)}...")
                img = cv2.imread(fname)
                if img is None: continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if gray_shape is None:
                    gray_shape = gray.shape[::-1]

                # --- LOGIC CORRECTION IS HERE ---
                image_for_refinement = gray # By default, use the original gray image

                find_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
                ret, corners = cv2.findChessboardCorners(gray, (self.cols, self.rows), find_flags)

                if not ret:
                    self.progress.emit(int(50 * (i + 1) / total_images), f"Standard search failed. Trying enhanced contrast...")
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray_enhanced = clahe.apply(gray)
                    find_flags_thorough = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
                    ret, corners = cv2.findChessboardCorners(gray_enhanced, (self.cols, self.rows), find_flags_thorough)
                    
                    if ret:
                        # If the enhanced image worked, use it for refinement
                        image_for_refinement = gray_enhanced

                if ret:
                    found_count += 1
                    objpoints.append(objp)
                    # Use the correct image (original or enhanced) for sub-pixel refinement
                    corners2 = cv2.cornerSubPix(image_for_refinement, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
                    self.corners_found.emit(fname, corners2)
                # --- END OF LOGIC CORRECTION ---

            if found_count == 0:
                self.finished.emit(False, "Could not find checkerboard corners in any images.", None, None, -1)
                return

            self.progress.emit(60, f"Found corners in {found_count}/{total_images} images. Calibrating...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

            if not ret:
                self.finished.emit(False, "Camera calibration failed.", None, None, -1)
                return

            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            reprojection_error = mean_error / len(objpoints)

            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.reprojection_error = reprojection_error

            self.progress.emit(100, "Calibration successful!")
            self.finished.emit(True, f"Calibration successful with {found_count} images.", mtx, dist, reprojection_error)
        except Exception as e:
            self.finished.emit(False, f"An error occurred: {str(e)}", None, None, -1)

# -------------------- GUI Tabs --------------------
class FrameExtractorTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.video_path = None
        self.folder_path = None
        self.image_file_paths = []
        self.extracted_frames_dir = None
        self.original_adjustment_image = None
        self.preview_files = []
        self.current_preview_index = 0
        self.thread = None
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        if hasattr(self.main_window, '_splitters') and isinstance(self.main_window._splitters, list):
            self.main_window._splitters.append(splitter)

        # Left Panel (Settings)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)

        # Step 1: Input Source
        source_group = QGroupBox("Step 1: Select Input Source")
        source_layout = QVBoxLayout(source_group)
        self.source_video_rb = QRadioButton("Extract from Video File")
        self.source_folder_rb = QRadioButton("Process Existing Image Files")
        self.source_video_rb.setChecked(True)
        source_layout.addWidget(self.source_video_rb)
        source_layout.addWidget(self.source_folder_rb)

        self.source_stack = QStackedWidget()
        video_widget = QWidget()
        video_layout = QFormLayout(video_widget)
        self.select_video_btn = QPushButton("Choose Video File")
        self.video_path_label = QLabel("No video selected.")
        video_layout.addRow(self.select_video_btn)
        video_layout.addRow(self.video_path_label)
        self.source_stack.addWidget(video_widget)

        folder_widget = QWidget()
        folder_layout = QFormLayout(folder_widget)
        self.select_files_btn = QPushButton("Choose Image Files")
        self.files_path_label = QLabel("No files selected.")
        folder_layout.addRow(self.select_files_btn)
        folder_layout.addRow(self.files_path_label)
        self.source_stack.addWidget(folder_widget)

        source_layout.addWidget(self.source_stack)
        left_layout.addWidget(source_group)

        # Step 2: Extraction Method (for video only)
        self.method_group = QGroupBox("Step 2: Frame Extraction Method")
        method_layout = QVBoxLayout(self.method_group)
        self.method_interval_rb = QRadioButton("By time interval (seconds)")
        self.method_specific_rb = QRadioButton("Specific frames (e.g., 0, 100, 200)")
        self.method_range_rb = QRadioButton("Frame range (start, end, step)")
        self.method_interval_rb.setChecked(True)
        self.method_interval_rb.toggled.connect(self.update_extractor_options)
        self.method_specific_rb.toggled.connect(self.update_extractor_options)
        self.method_range_rb.toggled.connect(self.update_extractor_options)
        method_layout.addWidget(self.method_interval_rb)
        method_layout.addWidget(self.method_specific_rb)
        method_layout.addWidget(self.method_range_rb)
        self.extractor_options_widget = QWidget()
        self.extractor_options_layout = QFormLayout(self.extractor_options_widget)
        method_layout.addWidget(self.extractor_options_widget)
        left_layout.addWidget(self.method_group)

        # Step 3: Enhancement Options
        post_proc_group = QGroupBox("Step 3: Enhancement Options")
        post_proc_layout = QVBoxLayout()
        self.grayscale_cb = QCheckBox("Convert to Grayscale")
        self.auto_contrast_cb = QCheckBox("Auto-Contrast Enhancement (Grayscale only)")
        self.enhance_cb = QCheckBox("Auto enhance frames (Color only)")
        self.rembg_cb = QCheckBox("Remove background")
        if not REMBG_AVAILABLE:
            self.rembg_cb.setDisabled(True)
            self.rembg_cb.setText("Remove background (rembg not installed)")
            self.rembg_cb.setToolTip("Install with: pip install rembg")

        post_proc_layout.addWidget(self.grayscale_cb)
        post_proc_layout.addWidget(self.auto_contrast_cb)
        post_proc_layout.addWidget(self.enhance_cb)
        post_proc_layout.addWidget(self.rembg_cb)
        post_proc_group.setLayout(post_proc_layout)
        left_layout.addWidget(post_proc_group)

        self.auto_contrast_cb.setEnabled(self.grayscale_cb.isChecked())
        self.grayscale_cb.toggled.connect(self.auto_contrast_cb.setEnabled)

        # Step 4: Start Button & Progress
        self.start_processing_btn = QPushButton("Start Processing")
        self.start_processing_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        left_layout.addWidget(self.start_processing_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Step 5: Live Image Adjustments
        self.adjustment_group = QGroupBox("Step 5: Live Image Adjustments")
        adj_layout = QVBoxLayout(self.adjustment_group)

        # --- Navigation ---
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("<< Previous")
        self.next_btn = QPushButton("Next >>")
        self.preview_info_label = QLabel("Previewing: N/A")
        self.preview_info_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.preview_info_label, 1)
        nav_layout.addWidget(self.next_btn)
        adj_layout.addLayout(nav_layout)

        # --- Sliders ---
        adj_form_layout = QFormLayout()

        # Brightness Slider
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 200) # 0.0 to 2.0
        self.brightness_slider.setValue(100)
        self.brightness_label = QLabel("1.00")
        brightness_box = QHBoxLayout()
        brightness_box.addWidget(self.brightness_slider)
        brightness_box.addWidget(self.brightness_label)
        adj_form_layout.addRow("Brightness:", brightness_box)

        # Contrast Slider
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200) # 0.0 to 2.0
        self.contrast_slider.setValue(100)
        self.contrast_label = QLabel("1.00")
        contrast_box = QHBoxLayout()
        contrast_box.addWidget(self.contrast_slider)
        contrast_box.addWidget(self.contrast_label)
        adj_form_layout.addRow("Contrast:", contrast_box)

        # Sharpness Slider
        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider.setRange(0, 200) # 0.0 to 2.0
        self.sharpness_slider.setValue(100)
        self.sharpness_label = QLabel("1.00")
        sharpness_box = QHBoxLayout()
        sharpness_box.addWidget(self.sharpness_slider)
        sharpness_box.addWidget(self.sharpness_label)
        adj_form_layout.addRow("Sharpness:", sharpness_box)

        adj_layout.addLayout(adj_form_layout)

        # --- Action Buttons ---
        adj_btn_layout = QHBoxLayout()
        self.reset_adj_btn = QPushButton("Reset Adjustments")
        self.apply_all_btn = QPushButton("Apply to All Images")
        self.apply_all_btn.setStyleSheet("background-color: #007BFF; color: white;")
        adj_btn_layout.addWidget(self.reset_adj_btn)
        adj_btn_layout.addWidget(self.apply_all_btn)
        adj_layout.addLayout(adj_btn_layout)

        # Add the group to the main layout and disable it initially
        self.adjustment_group.setEnabled(False)
        left_layout.addWidget(self.adjustment_group)
        left_layout.addStretch()

        # Right Panel (Preview & Log)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.preview_label = FitLabel("Preview will appear after processing")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #333; color: white; border: 1px solid #555;")

        log_group = QGroupBox("Operation Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)

        right_splitter = QSplitter(Qt.Vertical)
        if hasattr(self.main_window, '_splitters') and isinstance(self.main_window._splitters, list):
            self.main_window._splitters.append(right_splitter)
        right_splitter.addWidget(self.preview_label)
        right_splitter.addWidget(log_group)
        right_splitter.setSizes([400, 200])
        right_layout.addWidget(right_splitter)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([350, 650])
        main_layout.addWidget(splitter)

        # --- Connections ---
        self.source_video_rb.toggled.connect(self.update_source_selection)
        self.select_video_btn.clicked.connect(self.select_video)
        self.select_files_btn.clicked.connect(self.select_files)
        self.start_processing_btn.clicked.connect(self.start_processing)

        self.update_extractor_options()
        self.update_source_selection()

        self.brightness_slider.valueChanged.connect(self.update_adjustment_preview)
        self.contrast_slider.valueChanged.connect(self.update_adjustment_preview)
        self.sharpness_slider.valueChanged.connect(self.update_adjustment_preview)
        self.prev_btn.clicked.connect(self.show_previous_image)
        self.next_btn.clicked.connect(self.show_next_image)
        self.reset_adj_btn.clicked.connect(self.reset_adjustments)
        self.apply_all_btn.clicked.connect(self.apply_all_adjustments)

    def log_message(self, msg):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {msg}")
        QApplication.processEvents()

    def update_progress(self, percentage, message):
        self.progress_bar.setValue(percentage)
        self.log_message(message)

    @Slot()
    def update_source_selection(self):
        is_video = self.source_video_rb.isChecked()
        self.source_stack.setCurrentIndex(0 if is_video else 1)
        self.method_group.setEnabled(is_video)

    @Slot()
    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.video_path = path
            self.video_path_label.setText(os.path.basename(path))
            self.log_message(f"Selected video: {path}")

    @Slot()
    def select_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Image Files", "", "Image Files (*.png *.jpg *.jpeg)")
        if paths:
            self.image_file_paths = paths
            self.files_path_label.setText(f"{len(paths)} file(s) selected.")
            self.log_message(f"Selected {len(paths)} image file(s).")

    @Slot()
    def update_extractor_options(self):
        while self.extractor_options_layout.count():
            item = self.extractor_options_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        if self.method_interval_rb.isChecked():
            self.interval_le = QLineEdit("1.0")
            self.extractor_options_layout.addRow("Interval (seconds):", self.interval_le)
        elif self.method_specific_rb.isChecked():
            self.frames_le = QLineEdit("0, 100, 200")
            self.extractor_options_layout.addRow("Frame numbers:", self.frames_le)
        elif self.method_range_rb.isChecked():
            self.start_frame_le = QLineEdit("0")
            self.end_frame_le = QLineEdit("100")
            self.step_frame_le = QLineEdit("10")
            self.extractor_options_layout.addRow("Start frame:", self.start_frame_le)
            self.extractor_options_layout.addRow("End frame:", self.end_frame_le)
            self.extractor_options_layout.addRow("Step:", self.step_frame_le)

    @Slot()
    def start_processing(self):
        # Check for rembg download if selected
        if self.rembg_cb.isChecked() and not check_rembg_model():
            return # User canceled download or rembg is not available

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder for Results")
        if not output_dir: return

        params = {
            "output_dir": output_dir,
            "grayscale": self.grayscale_cb.isChecked(),
            "auto_contrast": self.auto_contrast_cb.isChecked(),
            "enhance": self.enhance_cb.isChecked(),
            "rembg": self.rembg_cb.isChecked(),
        }

        if self.source_video_rb.isChecked():
            if not self.video_path:
                QMessageBox.critical(self, "Error", "Please select a video file first.")
                return
            params["source_mode"] = "video"
            params["video_path"] = self.video_path
            if self.method_interval_rb.isChecked():
                params["method"] = "Interval"
                params["interval"] = safe_float(self.interval_le.text(), 1.0)
            elif self.method_specific_rb.isChecked():
                params["method"] = "Specific"
                params["frames"] = self.frames_le.text()
            elif self.method_range_rb.isChecked():
                params["method"] = "Range"
                params["start_frame"] = safe_int(self.start_frame_le.text())
                params["end_frame"] = safe_int(self.end_frame_le.text())
                params["step_frame"] = safe_int(self.step_frame_le.text(), 1)
        else:
            if not self.image_file_paths:
                QMessageBox.critical(self, "Error", "Please select one or more image files first.")
                return
            params["source_mode"] = "files"
            params["image_paths"] = self.image_file_paths

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.start_processing_btn.setEnabled(False)
        self.adjustment_group.setEnabled(False)

        self.thread = QThread()
        self.worker = Worker(self._run_processing_task, params)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(lambda err: QMessageBox.critical(self, "Processing Error", err))
        self.worker.result.connect(self.on_processing_complete)
        self.thread.start()
        self.thread.finished.connect(lambda: self.start_processing_btn.setEnabled(True))

    def on_processing_complete(self, result):
        self.log_message("Processing finished successfully!")
        self.progress_bar.setValue(100)
        self.extracted_frames_dir = result.get("final_folder")
        if self.extracted_frames_dir:
            self.preview_files = sorted([f for f in os.listdir(self.extracted_frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))], key=natural_key)
            if self.preview_files:
                self.adjustment_group.setEnabled(True)
                self.load_image_for_adjustment(0)
                QMessageBox.information(self, "Success", "Frames processed. You can now make live adjustments.")
            else:
                QMessageBox.information(self, "Success", "Processing complete, but no images found in the output folder.")
        else:
            QMessageBox.warning(self, "Finished", "Processing finished, but no output folder was returned.")

    def load_image_for_adjustment(self, index):
        if not self.preview_files or not (0 <= index < len(self.preview_files)):
            return

        self.current_preview_index = index
        filename = self.preview_files[index]
        image_path = os.path.join(self.extracted_frames_dir, filename)

        try:
            self.original_adjustment_image = Image.open(image_path).convert("RGB")
            self.preview_info_label.setText(f"Previewing: {filename} ({index + 1}/{len(self.preview_files)})")
            self.reset_adjustments() # Reset sliders for new image
        except Exception as e:
            self.log_message(f"Error loading {filename} for adjustment: {e}")
            self.preview_label.setText(f"Could not load:\n{filename}")

    def load_images_from_paths(self, new_paths):
        """Public method to reload the preview list with a new set of images."""
        if not new_paths:
            return

        self.extracted_frames_dir = os.path.dirname(new_paths[0])
        self.preview_files = sorted([os.path.basename(p) for p in new_paths], key=natural_key)

        if self.preview_files:
            self.adjustment_group.setEnabled(True)
            self.load_image_for_adjustment(0)
            self.log_message(f"Loaded {len(self.preview_files)} calibrated images into view.")
            QMessageBox.information(self, "Image Set Updated", "The image set has been updated with the calibrated versions.")
        else:
            self.adjustment_group.setEnabled(False)
            self.preview_label.setText("No images to display.")
            self.preview_info_label.setText("Previewing: N/A")

    @Slot()
    def update_adjustment_preview(self, _=None):
        if self.original_adjustment_image is None:
            return

        brightness = self.brightness_slider.value() / 100.0
        contrast = self.contrast_slider.value() / 100.0
        sharpness = self.sharpness_slider.value() / 100.0

        self.brightness_label.setText(f"{brightness:.2f}")
        self.contrast_label.setText(f"{contrast:.2f}")
        self.sharpness_label.setText(f"{sharpness:.2f}")

        try:
            img = self.original_adjustment_image
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpness)

            pixmap = pil_to_qpixmap(img)
            self.preview_label.setPixmap(pixmap) # Use FitLabel's scaling
        except Exception as e:
            self.log_message(f"Error applying adjustment: {e}")

    @Slot()
    def show_previous_image(self):
        if not self.preview_files: return
        new_index = (self.current_preview_index - 1 + len(self.preview_files)) % len(self.preview_files)
        self.load_image_for_adjustment(new_index)

    @Slot()
    def show_next_image(self):
        if not self.preview_files: return
        new_index = (self.current_preview_index + 1) % len(self.preview_files)
        self.load_image_for_adjustment(new_index)

    @Slot()
    def reset_adjustments(self):
        self.brightness_slider.setValue(100)
        self.contrast_slider.setValue(100)
        self.sharpness_slider.setValue(100)
        self.update_adjustment_preview() # Also update the view

    @Slot()
    def apply_all_adjustments(self):
        if not self.extracted_frames_dir or not self.preview_files:
            QMessageBox.warning(self, "Warning", "No images to apply adjustments to.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder for Adjusted Images")
        if not output_dir: return

        params = {
            "source_dir": self.extracted_frames_dir,
            "output_dir": output_dir,
            "image_files": self.preview_files,
            "brightness": self.brightness_slider.value() / 100.0,
            "contrast": self.contrast_slider.value() / 100.0,
            "sharpness": self.sharpness_slider.value() / 100.0,
        }

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.apply_all_btn.setEnabled(False)
        self.start_processing_btn.setEnabled(False)

        self.thread = QThread()
        self.worker = Worker(self._run_adjustment_task, params)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(lambda err: QMessageBox.critical(self, "Adjustment Error", err))
        self.worker.result.connect(lambda res: QMessageBox.information(self, "Success", f"Adjustments applied and saved in:\n{output_dir}"))
        self.thread.start()
        self.thread.finished.connect(lambda: (self.apply_all_btn.setEnabled(True), self.start_processing_btn.setEnabled(True)))

    @staticmethod
    def _run_adjustment_task(progress_callback, params):
        source_dir = params["source_dir"]
        output_dir = params["output_dir"]
        image_files = params["image_files"]
        brightness = params["brightness"]
        contrast = params["contrast"]
        sharpness = params["sharpness"]

        os.makedirs(output_dir, exist_ok=True)
        total_files = len(image_files)
        progress_callback(0, f"Starting adjustment for {total_files} images...")

        for i, filename in enumerate(image_files):
            try:
                img = Image.open(os.path.join(source_dir, filename))
                if img.mode != "RGB":
                    img = img.convert("RGB")

                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast)
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(sharpness)

                img.save(os.path.join(output_dir, filename))

                percentage = int((i + 1) / total_files * 100)
                progress_callback(percentage, f"Adjusted and saved {filename} ({i+1}/{total_files})")
            except Exception as e:
                print(f"Could not process {filename}: {e}") # Log to console

        progress_callback(100, "Adjustment process finished.")
        return {"success": True}

    @staticmethod
    def _run_processing_task(progress_callback, params):
        source_mode = params["source_mode"]
        output_dir = params["output_dir"]

        raw_folder = os.path.join(output_dir, "raw_frames")
        os.makedirs(raw_folder, exist_ok=True)

        # --- Initial Extraction / Copying ---
        if source_mode == "video":
            progress_callback(0, "Starting video extraction...")
            video_path = params["video_path"]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): raise IOError("Error: Failed to open video file.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_callback(0, f"Video info: {total_frames} frames, {fps:.2f} FPS")

            frame_numbers = []
            if params["method"] == 'Interval':
                interval_seconds = params.get("interval", 1.0)
                if interval_seconds > 0 and fps > 0:
                    interval_frames = max(1, int(fps * interval_seconds))
                    frame_numbers = list(range(0, total_frames, interval_frames))
                else:
                    frame_numbers = list(range(0, total_frames, 30))
            elif params["method"] == 'Specific':
                frame_numbers = sorted(list(set([safe_int(f.strip()) for f in params["frames"].split(',')])))
            elif params["method"] == 'Range':
                frame_numbers = list(range(params["start_frame"], params["end_frame"] + 1, max(1, params["step_frame"])))

            total_to_extract = len(frame_numbers)
            progress_callback(1, f"Identified {total_to_extract} frames to extract.")

            for i, fn in enumerate(frame_numbers):
                if 0 <= fn < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(os.path.join(raw_folder, f"frame_{fn:05d}.png"), frame)

                percentage = int((i + 1) / total_to_extract * 40)
                progress_callback(percentage, f"Extracting frame {fn} ({i+1}/{total_to_extract})")

            cap.release()
        else: # source_mode == "files"
            progress_callback(1, "Copying selected images...")
            image_files = params["image_paths"]
            if not image_files:
                raise FileNotFoundError("No image files were selected.")
            for i, source_path in enumerate(image_files):
                filename = os.path.basename(source_path)
                shutil.copy(source_path, os.path.join(raw_folder, filename))
                percentage = int((i+1) / len(image_files) * 40)
                progress_callback(percentage, f"Copying {filename}")

        # --- Post-Processing Pipeline ---
        final_folder = raw_folder
        current_progress = 40

        active_steps = []
        if params.get("grayscale"): active_steps.append("grayscale")
        if params.get("enhance"): active_steps.append("enhance")
        if params.get("rembg") and REMBG_AVAILABLE: active_steps.append("rembg")

        if not active_steps:
            progress_callback(100, "Processing complete.")
            return {"final_folder": final_folder}

        progress_per_step = (100.0 - current_progress) / len(active_steps)

        # Step 1: Grayscale and Auto-Contrast
        if "grayscale" in active_steps:
            grayscale_folder = os.path.join(output_dir, "grayscale_frames")
            os.makedirs(grayscale_folder, exist_ok=True)
            image_files = sorted([f for f in os.listdir(final_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
            total_to_process = len(image_files)

            for i, filename in enumerate(image_files):
                img_path = os.path.join(final_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue

                if params.get("auto_contrast"):
                    img = cv2.equalizeHist(img)

                cv2.imwrite(os.path.join(grayscale_folder, filename), img)

                percentage = int(current_progress + ((i + 1) / total_to_process * progress_per_step))
                progress_callback(percentage, f"Applying grayscale/contrast: {filename} ({i+1}/{total_to_process})")

            final_folder = grayscale_folder
            current_progress += progress_per_step

        # Step 2: Auto Enhance (Sharpness/Contrast for Color)
        if "enhance" in active_steps:
            enhanced_folder = os.path.join(output_dir, "enhanced_frames")
            os.makedirs(enhanced_folder, exist_ok=True)
            image_files = sorted([f for f in os.listdir(final_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
            total_to_process = len(image_files)
            for i, filename in enumerate(image_files):
                img = Image.open(os.path.join(final_folder, filename))

                if img.mode not in ['L', '1']:
                    if img.mode == 'P': img = img.convert('RGB')
                    img_sharp = ImageEnhance.Sharpness(img).enhance(2.0)
                    img_contrast = ImageEnhance.Contrast(img_sharp).enhance(1.5)
                    img_contrast.save(os.path.join(enhanced_folder, filename))
                else:
                    shutil.copy(os.path.join(final_folder, filename), os.path.join(enhanced_folder, filename))

                percentage = int(current_progress + ((i + 1) / total_to_process * progress_per_step))
                progress_callback(percentage, f"Enhancing {filename} ({i+1}/{total_to_process})")
            final_folder = enhanced_folder
            current_progress += progress_per_step

        # Step 3: Remove Background
        if "rembg" in active_steps:
            rembg_folder = os.path.join(output_dir, "transparent_frames")
            os.makedirs(rembg_folder, exist_ok=True)
            rembg_sess = new_session()
            image_files = sorted([f for f in os.listdir(final_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
            total_to_process = len(image_files)
            for i, filename in enumerate(image_files):
                # Ensure output has a consistent .png extension for transparency
                output_filename = os.path.splitext(filename)[0] + ".png"
                with open(os.path.join(final_folder, filename), 'rb') as f_in, open(os.path.join(rembg_folder, output_filename), 'wb') as f_out:
                    f_out.write(remove(f_in.read(), session=rembg_sess))
                percentage = int(current_progress + ((i + 1) / total_to_process * progress_per_step))
                progress_callback(percentage, f"Removing background from {filename} ({i+1}/{total_to_process})")
            final_folder = rembg_folder

        progress_callback(100, "Processing finished.")
        return {"final_folder": final_folder}

# =============================================================================
# START: Final and Corrected ROITab Class
# =============================================================================

class ROITab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.roi_drawer = ROIDrawer()
        self.image_path = None
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Left Panel: Controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)

        # Input Group
        input_group = QGroupBox("1. Input Image")
        input_layout = QVBoxLayout(input_group)
        self.load_from_tab1_btn = QPushButton("Load from Frame Processor")
        self.load_standalone_btn = QPushButton("Load Standalone Image...")
        input_layout.addWidget(self.load_from_tab1_btn)
        input_layout.addWidget(self.load_standalone_btn)
        left_layout.addWidget(input_group)

        # Auto-Masking Group
        auto_mask_group = QGroupBox("2. Automatic Mask Generation")
        auto_mask_layout = QVBoxLayout(auto_mask_group)
        self.generate_mask_btn = QPushButton("Create Foreground Mask (rembg)")
        if not REMBG_AVAILABLE:
            self.generate_mask_btn.setText("Create Mask (rembg not installed)")
            self.generate_mask_btn.setToolTip("Install 'rembg' for this feature.")
            self.generate_mask_btn.setEnabled(False)
        auto_mask_layout.addWidget(self.generate_mask_btn)
        left_layout.addWidget(auto_mask_group)

        # ROI Management Group
        manage_group = QGroupBox("3. ROI Management")
        manage_layout = QHBoxLayout(manage_group)
        self.save_roi_btn = QPushButton("Save ROI Mask")
        self.load_roi_btn = QPushButton("Load ROI Mask")
        manage_layout.addWidget(self.save_roi_btn)
        manage_layout.addWidget(self.load_roi_btn)
        left_layout.addWidget(manage_group)

        # Layer Management
        layer_group = QGroupBox("4. Layer Management")
        layer_layout = QVBoxLayout(layer_group)
        self.layer_list = QListWidget()
        self.layer_list.itemClicked.connect(self.on_layer_selected)
        
        layer_btn_layout = QHBoxLayout()
        self.del_layer_btn = QPushButton("Delete Selected")
        self.clear_all_btn = QPushButton("Clear All")
        layer_btn_layout.addWidget(self.del_layer_btn)
        layer_btn_layout.addWidget(self.clear_all_btn)
        
        layer_layout.addWidget(self.layer_list)
        layer_layout.addLayout(layer_btn_layout)
        left_layout.addWidget(layer_group)
        left_layout.addStretch()

        # Right Panel: ROI Drawer + Toolbar
        right_group = QGroupBox("ROI Editor")
        right_layout = QVBoxLayout(right_group)

        toolbar = QToolBar("ROI Tools")
        self.tool_actions = {}
        tool_data = [
            ("rect_add", "Add Rectangle"), ("rect_sub", "Subtract Rectangle"),
            ("ellipse_add", "Add Ellipse"), ("ellipse_sub", "Subtract Ellipse"),
            ("brush_add", "Add Brush"), ("brush_sub", "Subtract Brush"),
            ("pan", "Pan/Zoom Tool")
        ]

        for key, text in tool_data:
            action = QAction(text, self, checkable=True)
            shape, mode = key.split('_') if '_' in key else (key, 'add')
            # CHANGE IS HERE: added '=False' to 'checked'
            action.triggered.connect(lambda checked=False, s=shape, m=mode: self.select_tool(s, m))
            self.tool_actions[key] = action
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Brush Size:"))
        self.brush_size_spinbox = QSpinBox()
        self.brush_size_spinbox.setRange(1, 200)
        self.brush_size_spinbox.setValue(20)
        self.brush_size_spinbox.valueChanged.connect(self.roi_drawer.set_brush_size)
        toolbar.addWidget(self.brush_size_spinbox)
        
        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.roi_drawer, 1)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_group)
        splitter.setSizes([320, 700])
        layout.addWidget(splitter)
        if hasattr(self.main_window, '_splitters') and isinstance(self.main_window._splitters, list):
            self.main_window._splitters.append(splitter)

        # Connections
        self.load_from_tab1_btn.clicked.connect(self.load_image_from_tab1)
        self.load_standalone_btn.clicked.connect(self.load_standalone_image)
        self.generate_mask_btn.clicked.connect(self.generate_auto_mask)
        self.save_roi_btn.clicked.connect(self.save_roi)
        self.load_roi_btn.clicked.connect(self.load_roi)
        self.del_layer_btn.clicked.connect(lambda: self.roi_drawer.remove_layer(self.layer_list.currentRow()))
        self.clear_all_btn.clicked.connect(self.roi_drawer.clear_all)
        self.roi_drawer.layers_changed.connect(self.update_layer_list)

        self.select_tool("rect", "add") # Set default tool

    def select_tool(self, shape, mode):
        self.roi_drawer.set_mode(shape, mode)
        key = f"{shape}_{mode}" if '_' not in shape else shape
        for action_key, action in self.tool_actions.items():
            action.setChecked(action_key == key)

    @Slot(list)
    def update_layer_list(self, names):
        self.layer_list.clear()
        self.layer_list.addItems(names)
        if self.roi_drawer.selected_layer_idx is not None and 0 <= self.roi_drawer.selected_layer_idx < self.layer_list.count():
            self.layer_list.setCurrentRow(self.roi_drawer.selected_layer_idx)

    @Slot(QListWidgetItem)
    def on_layer_selected(self, item):
        if item:
            row = self.layer_list.row(item)
            self.roi_drawer.selected_layer_idx = row
            self.roi_drawer.update()

    def set_background_image(self, image_path):
        if image_path and os.path.exists(image_path):
            self.image_path = image_path
            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None: raise Exception("Failed to load image with OpenCV")
                self.roi_drawer.set_image(img)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load or process the image: {e}")
        else:
            QMessageBox.warning(self, "Warning", "Could not find a valid reference image.")

    @Slot()
    def load_image_from_tab1(self):
        ref_image_path = self.main_window.get_reference_image_path()
        if ref_image_path:
            self.set_background_image(ref_image_path)
        else:
            QMessageBox.warning(self, "Warning", "No image processed in 'Frame Processor' tab. Please process images or load a standalone image.")

    @Slot()
    def load_standalone_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Image for ROI", "", "Image Files (*.png *.jpg *.jpeg *.tif)")
        if path:
            self.set_background_image(path)

    @Slot()
    def generate_auto_mask(self):
        if self.roi_drawer.img_np is None:
            QMessageBox.warning(self, "Warning", "Please load an image before generating a mask.")
            return
        if not check_rembg_model(): return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if not self.image_path:
                raise Exception("Image path is not available. Please load the image again.")
            
            with open(self.image_path, 'rb') as f_in:
                input_bytes = f_in.read()

            session = new_session("u2net")
            output_bytes = remove(input_bytes, session=session)

            img_rgba = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
            alpha = np.array(img_rgba)[:, :, 3]
            mask = (alpha > 50)

            self.roi_drawer.layers.append({
                "type": "imported",
                "mode": "add",
                "mask": mask
            })
            self.roi_drawer._recomposite_mask()
            QMessageBox.information(self, "Success", "Automatic mask generated and added as a new layer.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate mask: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def get_roi_mask(self):
        return self.roi_drawer.final_mask

    @Slot()
    def save_roi(self):
        mask = self.get_roi_mask()
        if mask is None or not mask.any():
            QMessageBox.warning(self, "Warning", "No ROI mask to save (mask is empty).")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save ROI Mask", "", "PNG Files (*.png)")
        if path:
            try:
                mask_image = (mask * 255).astype(np.uint8)
                imageio.imwrite(path, mask_image)
                QMessageBox.information(self, "Success", f"ROI mask saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save ROI: {e}")

    @Slot()
    def load_roi(self):
        if self.roi_drawer.img_np is None:
            QMessageBox.warning(self, "Warning", "Please load a background image first before loading an ROI mask.")
            return

        path, _ = QFileDialog.getOpenFileName(self, "Load ROI Mask", "", "Image Files (*.png *.jpg *.bmp *.tif)")
        if path:
            try:
                mask_img = imageio.imread(path)
                mask = mask_img > 128
                
                if mask.shape != self.roi_drawer.img_np.shape:
                     QMessageBox.warning(self, "Shape Mismatch", "The loaded mask dimensions do not match the background image.")
                     return

                self.roi_drawer.layers.append({
                    "type": "imported",
                    "mode": "add",
                    "mask": mask
                })
                self.roi_drawer._recomposite_mask()
                QMessageBox.information(self, "Success", f"ROI mask loaded as a new layer from {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load ROI: {e}")

# =============================================================================
# END: Final and Corrected ROITab Class
# =============================================================================
# =============================================================================
# TAB 3: NcorrConfigTab (REPLACED with the old, more functional version)
# =============================================================================
class NcorrConfigTab(QWidget):
    """
    Builds Ncorr-friendly DIC configuration presets from a loaded sample image.
    Features a preview with a 2-point click calibration for units-per-pixel.
    Produces five presets and lets the user save the selected one as a structured JSON.
    """
    def __init__(self, main_window=None):
        super().__init__(parent=main_window if isinstance(main_window, QWidget) else None)
        self.main_window = main_window
        self.sample_image_path = None
        self.original_pixmap = None
        self.presets = {}
        self.calib_points = [] # Stores the (x,y) of the two clicked points in original image coords
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # Top selection
        top_frame = QWidget()
        top_layout = QHBoxLayout(top_frame)
        select_btn = QPushButton("Choose Sample Image")
        select_btn.clicked.connect(self.select_sample_image)
        self.sample_path_label = QLabel("No image selected")
        top_layout.addWidget(select_btn)
        top_layout.addWidget(self.sample_path_label, 1)
        main_layout.addWidget(top_frame)

        # Main splitter: Image on the left, controls on the right
        main_splitter = QSplitter(Qt.Horizontal)
        if hasattr(self.main_window, '_splitters') and isinstance(self.main_window._splitters, list):
            self.main_window._splitters.append(main_splitter)

        # Left Panel: Image Preview and Calibration
        preview_calib_group = QGroupBox("Image Preview & Calibration")
        preview_calib_layout = QVBoxLayout(preview_calib_group)

        self.preview_label = ClickableLabel("Select an image to see preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #333; color: white; border: 1px solid #555;")
        self.preview_label.setMinimumSize(0, 0)
        self.preview_label.clicked.connect(self.handle_image_click)
        preview_calib_layout.addWidget(self.preview_label, 1)

        calib_form_layout = QFormLayout()
        self.known_dist_le = QLineEdit("1.0")
        self.calc_upp_btn = QPushButton("Calculate from 2 points")
        self.calc_upp_btn.setToolTip("Click two points on the image, enter the known distance, then click here.")
        self.calc_upp_btn.clicked.connect(self.calculate_units_per_pixel)
        self.calc_upp_btn.setEnabled(False) # Enabled once 2 points are clicked

        calib_form_layout.addRow("Known distance between points:", self.known_dist_le)
        calib_form_layout.addRow(self.calc_upp_btn)
        preview_calib_layout.addLayout(calib_form_layout)


        # Right Panel: Presets and Parameters
        right_splitter = QSplitter(Qt.Vertical)
        if hasattr(self.main_window, '_splitters') and isinstance(self.main_window._splitters, list):
            self.main_window._splitters.append(right_splitter)

        preset_group = QGroupBox("Presets (auto-derived from sample)")
        preset_layout = QVBoxLayout()
        self.rb_fast = QRadioButton("Fast")
        self.rb_balanced = QRadioButton("Balanced")
        self.rb_accurate = QRadioButton("Accurate")
        self.rb_low_texture = QRadioButton("Robust (Low Texture)")
        self.rb_high_gradient = QRadioButton("High Strain Gradient")
        for rb in [self.rb_fast, self.rb_balanced, self.rb_accurate, self.rb_low_texture, self.rb_high_gradient]:
            rb.toggled.connect(self.update_preset_display)
            preset_layout.addWidget(rb)
        preset_layout.addStretch()
        preset_group.setLayout(preset_layout)

        params_group = QGroupBox("Ncorr DIC Parameters")
        params_layout = QVBoxLayout()
        self.params_text = QTextEdit()
        self.params_text.setReadOnly(True)
        self.params_text.setFontFamily("Courier")
        form = QFormLayout()
        self.units_le = QLineEdit("mm")
        self.upp_le = QLineEdit("0.0")  # units per pixel
        self.upp_le.setReadOnly(True) # Make it read-only, populated by calculation
        form.addRow("Units (e.g., mm):", self.units_le)
        form.addRow("Units per pixel:", self.upp_le)
        params_layout.addLayout(form)
        params_layout.addWidget(self.params_text)
        params_group.setLayout(params_layout)

        right_splitter.addWidget(preset_group)
        right_splitter.addWidget(params_group)
        right_splitter.setSizes([200, 400])

        main_splitter.addWidget(preview_calib_group)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([600, 400])
        main_layout.addWidget(main_splitter)

        # Save Button
        save_btn = QPushButton("Save Selected Preset (JSON)")
        save_btn.clicked.connect(self.save_selected_preset)
        main_layout.addWidget(save_btn)

        self.rb_balanced.setChecked(True)

    @Slot()
    def select_sample_image(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Select Sample Image", "", "Image Files (*.png *.jpg *.jpeg *.tif *.bmp)")
        if path:
            self.sample_image_path = path
            self.sample_path_label.setText(os.path.basename(path))
            self.original_pixmap = QPixmap(path)
            self.calib_points = [] # Reset points on new image
            self.calc_upp_btn.setEnabled(False)
            self.update_preview_display()
            self.generate_presets_from_image(path)
            self.rb_balanced.setChecked(True)
            self.update_preset_display()

    def update_preview_display(self):
        """
        Updates the preview label with the original pixmap, drawing points and a line on a
        full-resolution copy before scaling it to fit the label.
        """
        if not self.original_pixmap or self.original_pixmap.isNull():
            self.preview_label.setPixmap(QPixmap()) # Clear the label
            return

        display_pixmap = self.original_pixmap.copy()

        if self.calib_points:
            painter = QPainter(display_pixmap)
            pen_width = max(2, self.original_pixmap.width() // 300)
            pen = QPen(Qt.red, pen_width)
            painter.setPen(pen)

            if len(self.calib_points) == 2:
                painter.drawLine(self.calib_points[0], self.calib_points[1])

            radius = pen_width * 2
            for point in self.calib_points:
                painter.drawEllipse(point, radius, radius)
            painter.end()

        scaled_pixmap = display_pixmap.scaled(
            self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)

    @Slot(QPoint)
    def handle_image_click(self, label_pos):
        """Handles a click on the preview label, converting it to image coordinates."""
        if not self.original_pixmap:
            return

        scaled_pixmap = self.preview_label.pixmap()
        if not scaled_pixmap or scaled_pixmap.isNull():
            return

        # Account for alignment (letterboxing/pillarboxing)
        x_offset = (self.preview_label.width() - scaled_pixmap.width()) / 2
        y_offset = (self.preview_label.height() - scaled_pixmap.height()) / 2

        pixmap_x = label_pos.x() - x_offset
        pixmap_y = label_pos.y() - y_offset

        # Convert scaled pixmap coordinates to original image coordinates
        if scaled_pixmap.width() == 0 or scaled_pixmap.height() == 0: return
        scale_x = self.original_pixmap.width() / scaled_pixmap.width()
        scale_y = self.original_pixmap.height() / scaled_pixmap.height()

        original_x = int(pixmap_x * scale_x)
        original_y = int(pixmap_y * scale_y)

        # Check if click is within the actual image area
        if not (0 <= original_x < self.original_pixmap.width() and 0 <= original_y < self.original_pixmap.height()):
            return # Click was in the padded area

        if len(self.calib_points) >= 2:
            self.calib_points = [] # Reset if we already have two points

        self.calib_points.append(QPoint(original_x, original_y))

        if len(self.calib_points) == 2:
            self.calc_upp_btn.setEnabled(True)
        else:
            self.calc_upp_btn.setEnabled(False)

        self.update_preview_display() # Redraw the preview with the new point

    @Slot()
    def calculate_units_per_pixel(self):
        if len(self.calib_points) != 2:
            QMessageBox.warning(self, "Warning", "Please select exactly two points on the image.")
            return

        p1 = self.calib_points[0]
        p2 = self.calib_points[1]

        pixel_dist = math.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
        known_dist = safe_float(self.known_dist_le.text(), 0.0)

        if pixel_dist < 1e-6:
            QMessageBox.critical(self, "Error", "The selected points are identical. Cannot calculate.")
            return
        if known_dist <= 0:
            QMessageBox.critical(self, "Error", "Known distance must be a positive number.")
            return

        upp = known_dist / pixel_dist
        self.upp_le.setText(f"{upp:.6f}")
        QMessageBox.information(self, "Success", f"Calculated Units Per Pixel: {upp:.6f}")

        if self.sample_image_path:
             self.generate_presets_from_image(self.sample_image_path)
             self.update_preset_display()


    def _analyze_image(self, img_gray):
        """Return texture and edge density metrics used to build presets."""
        lap = cv2.Laplacian(img_gray, cv2.CV_64F)
        texture = float(lap.var())
        v = np.median(img_gray)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(img_gray, lower, upper)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        return texture, edge_density

    def _suggest_radius(self, width, height, texture, edge_density):
        """
        Heuristic subset radius suggestion.
        """
        base = max(9, int(round(min(width, height) / 300.0)))  # scale with image size
        if texture < 50.0 or edge_density < 0.02:
            base += 6
        elif texture > 300.0 and edge_density > 0.05:
            base = max(7, base - 2)
        if base % 2 == 0:
            base += 1
        return base

    def _threads(self):
        try:
            return max(1, int(os.cpu_count() or 1))
        except Exception:
            return 1

    def _make_cfg(self, *, radius, interp, subregion, threads, dic_config, debug, units, units_per_pixel):
        """Builds the nested dictionary structure for the Ncorr config."""
        return {
            "dic": {
                "interp": str(interp),
                "subregion": str(subregion),
                "radius": int(radius),
                "threads": int(threads),
                "dic_config": str(dic_config),
                "debug": bool(debug),
                "spacing": 1,
                "cutoff_corrcoef": 0.001,
                "perspective": "LAGRANGIAN",
                "units": str(units),
                "units_per_pixel": float(units_per_pixel),
                "lenscoef": 0.0
            },
            "strain": {
                "radius": int(radius),
                "perspective": "LAGRANGIAN"
            }
        }

    def generate_presets_from_image(self, path):
        if path and os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                QMessageBox.critical(self, "Error", "Failed to load the sample image.")
                return
            h, w = img.shape[:2]
            texture, edge_density = self._analyze_image(img)
            base_radius = self._suggest_radius(w, h, texture, edge_density)
        else:
            base_radius = 21

        tcount = self._threads()
        units = self.units_le.text().strip() or "mm"
        units_per_pixel = safe_float(self.upp_le.text(), 0.0)

        self.presets = {
            "Fast": self._make_cfg(
                radius=max(7, base_radius - 2), interp="CUBIC_KEYS_PRECOMPUTE", subregion="CIRCLE",
                threads=max(1, tcount // 2), dic_config="KEEP_MOST_POINTS", debug=False,
                units=units, units_per_pixel=units_per_pixel,
            ),
            "Balanced": self._make_cfg(
                radius=base_radius, interp="QUINTIC_BSPLINE_PRECOMPUTE", subregion="CIRCLE",
                threads=tcount, dic_config="KEEP_MOST_POINTS", debug=False,
                units=units, units_per_pixel=units_per_pixel,
            ),
            "Accurate": self._make_cfg(
                radius=base_radius + 2, interp="QUINTIC_BSPLINE_PRECOMPUTE", subregion="SQUARE",
                threads=max(1, tcount - 1), dic_config="STRICT_CONVERGENCE", debug=False,
                units=units, units_per_pixel=units_per_pixel,
            ),
            "Robust (Low Texture)": self._make_cfg(
                radius=base_radius + 6, interp="QUINTIC_BSPLINE_PRECOMPUTE", subregion="CIRCLE",
                threads=tcount, dic_config="KEEP_MOST_POINTS", debug=True,
                units=units, units_per_pixel=units_per_pixel,
            ),
            "High Strain Gradient": self._make_cfg(
                radius=max(7, base_radius - 2), interp="QUINTIC_BSPLINE_PRECOMPUTE", subregion="SQUARE",
                threads=tcount, dic_config="STRICT_CONVERGENCE", debug=True,
                units=units, units_per_pixel=units_per_pixel,
            ),
        }

    @Slot()
    def update_preset_display(self):
        if not self.presets:
            self.params_text.setPlainText("Load a sample image to generate presets.")
            return

        name = self._current_preset_name()
        cfg = self.presets.get(name, {})
        if not cfg:
            self.params_text.setPlainText("Preset not found.")
            return

        display_text = f"Preset: {name}\n\n"
        display_text += json.dumps(cfg, indent=2)
        self.params_text.setPlainText(display_text)

    def _current_preset_name(self):
        if self.rb_fast.isChecked(): return "Fast"
        if self.rb_balanced.isChecked(): return "Balanced"
        if self.rb_accurate.isChecked(): return "Accurate"
        if self.rb_low_texture.isChecked(): return "Robust (Low Texture)"
        if self.rb_high_gradient.isChecked(): return "High Strain Gradient"
        return "Balanced"

    @Slot()
    def save_selected_preset(self):
        if not self.presets:
            QMessageBox.warning(self, "Warning", "No presets available. Please load a sample image first.")
            return
        name = self._current_preset_name()
        cfg = self.presets.get(name)
        if not cfg:
            QMessageBox.warning(self, "Warning", "Please choose a preset to save.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Ncorr Preset", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            QMessageBox.information(self, "Success", f"Preset saved to: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save preset: {e}")

    def get_config(self):
        """Returns the currently displayed configuration as a dictionary."""
        name = self._current_preset_name()
        if not name or not self.presets:
            return {}
        return self.presets.get(name, {}).get("dic", {})

# -------------------- Image Quality Assessment Tab (ENHANCED) --------------------
class QualityAssessmentTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        top_frame = QWidget()
        top_layout = QHBoxLayout(top_frame)
        analyze_btn = QPushButton("Choose Image for Analysis")
        analyze_btn.clicked.connect(self.analyze_image_quality)
        self.quality_image_label = QLabel("No image selected")
        top_layout.addWidget(analyze_btn)
        top_layout.addWidget(self.quality_image_label, 1)
        main_layout.addWidget(top_frame)

        content_splitter = QSplitter(Qt.Horizontal)
        if hasattr(self.main_window, '_splitters') and isinstance(self.main_window._splitters, list):
            self.main_window._splitters.append(content_splitter)

        preview_group = QGroupBox("Image Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = FitLabel("Image will appear here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #333; color: white;")
        preview_layout.addWidget(self.preview_label)

        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)

        # --- Basic Stats ---
        basic_stats_group = QGroupBox("Basic Metrics")
        basic_layout = QFormLayout(basic_stats_group)
        self.mean_label = QLabel("N/A")
        self.std_label = QLabel("N/A")
        self.entropy_label = QLabel("N/A")
        basic_layout.addRow("Mean Brightness:", self.mean_label)
        basic_layout.addRow("Std. Dev. (Contrast):", self.std_label)
        basic_layout.addRow("Shannon Entropy:", self.entropy_label)

        # --- Advanced DIC Metrics ---
        adv_stats_group = QGroupBox("DIC Quality Metrics")
        adv_layout = QFormLayout(adv_stats_group)
        self.mig_label = QLabel("N/A")
        self.speckle_size_label = QLabel("N/A")
        self.speckle_density_label = QLabel("N/A")
        adv_layout.addRow("Mean Intensity Gradient (MIG):", self.mig_label)
        adv_layout.addRow("Mean Speckle Size (pixels):", self.speckle_size_label)
        adv_layout.addRow("Speckle Density (%):", self.speckle_density_label)

        hist_group = QGroupBox("Grayscale Histogram")
        hist_layout = QVBoxLayout(hist_group)
        self.histogram_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.hist_ax = self.histogram_canvas.figure.add_subplot(111)
        hist_layout.addWidget(self.histogram_canvas)

        stats_layout.addWidget(basic_stats_group)
        stats_layout.addWidget(adv_stats_group)
        stats_layout.addWidget(hist_group)
        stats_layout.addStretch()

        content_splitter.addWidget(preview_group)
        content_splitter.addWidget(stats_widget)
        content_splitter.setSizes([500, 350])
        main_layout.addWidget(content_splitter, 1)

    @Slot()
    def analyze_image_quality(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Image Files (*.png *.jpg *.jpeg *.tif *.bmp)")
        if not path:
            return
        self.quality_image_label.setText(os.path.basename(path))
        try:
            pixmap = QPixmap(path)
            self.preview_label.setPixmap(pixmap)

            img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise ValueError("Could not load image in grayscale.")

            # 1. Basic Stats
            mean_val, std_val = cv2.meanStdDev(img_gray)
            self.mean_label.setText(f"{mean_val[0][0]:.2f}")
            self.std_label.setText(f"{std_val[0][0]:.2f}")

            # 2. Histogram and Entropy
            hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).ravel()
            self.hist_ax.clear()
            self.hist_ax.plot(hist, color='k')
            self.hist_ax.set_xlim([0, 256])
            self.hist_ax.set_title("Grayscale Distribution")
            self.hist_ax.set_xlabel("Pixel Value")
            self.hist_ax.set_ylabel("Frequency")
            self.histogram_canvas.figure.tight_layout()
            self.histogram_canvas.draw()

            # Calculate Shannon Entropy
            prob_dist = hist / hist.sum()
            shannon_entropy = entropy(prob_dist, base=2)
            self.entropy_label.setText(f"{shannon_entropy:.4f} bits/pixel")

            # 3. Mean Intensity Gradient (MIG)
            grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            mig = np.mean(grad_mag)
            self.mig_label.setText(f"{mig:.2f}")

            # 4. Speckle Analysis
            _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                mean_area = np.mean(areas)
                mean_diameter = 2 * np.sqrt(mean_area / np.pi)
                total_speckle_area = np.sum(areas)
                density = (total_speckle_area / img_gray.size) * 100
                self.speckle_size_label.setText(f"{mean_diameter:.2f}")
                self.speckle_density_label.setText(f"{density:.2f}%")
            else:
                self.speckle_size_label.setText("N/A (No speckles found)")
                self.speckle_density_label.setText("0.00%")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze image: {e}")


# =============================================================================
# TAB 5: PatternGeneratorTab (MERGED with old version)
# =============================================================================
class PatternMaster: # Brought from old code
    def __init__(self, width_mm, height_mm, dpi=300):
        if not all(isinstance(i, (int, float)) and i > 0 for i in [width_mm, height_mm, dpi]):
            raise ValueError("Width, height, and DPI values must be positive numbers.")
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.dpi = dpi
        self.pixels_per_mm = self.dpi / 25.4
        self.width_px = int(round(self.width_mm * self.pixels_per_mm))
        self.height_px = int(round(self.height_mm * self.pixels_per_mm))

    def _save_image_with_metadata(self, image_array, path):
        if not path.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff', '.png')):
            path += '.tiff'
        pil_image = Image.fromarray(image_array.astype(np.uint8))
        pil_image.save(path, dpi=(self.dpi, self.dpi))

    def generate_speckle_pattern(self, path, min_size_mm, max_size_mm, density, contrast, noise, blur_sigma=0.5):
        h, w = self.height_px, self.width_px
        image = np.ones((h, w), dtype=np.float32)
        min_speckle_px = max(2, int(round(min_size_mm * self.pixels_per_mm)))
        max_speckle_px = max(min_speckle_px, int(round(max_size_mm * self.pixels_per_mm)))
        avg_diameter_px = (min_speckle_px + max_speckle_px) / 2.0
        if avg_diameter_px == 0:
            return np.ones((h, w), dtype=np.uint8) * 255
        num_speckles = int(density * (w * h) / (avg_diameter_px ** 2))
        for _ in range(num_speckles):
            diameter = random.randint(min_speckle_px, max_speckle_px)
            if diameter < 2:
                continue
            speckle = self._create_single_speckle(diameter, blur_sigma)
            sh, sw = speckle.shape
            if (h - sh) <= 0 or (w - sw) <= 0:
                continue
            x, y = random.randint(0, w - sw), random.randint(0, h - sh)
            current_slice = image[y:y + sh, x:x + sw]
            image[y:y + sh, x:x + sw] = np.minimum(current_slice, 1 - speckle * contrast)
        if noise > 0:
            image += np.random.normal(0, noise / 255.0, image.shape)
        final_image = np.clip(image * 255, 0, 255)
        if path:
            self._save_image_with_metadata(final_image, path)
        return final_image

    def generate_voronoi_pattern(self, path, num_points):
        h, w = self.height_px, self.width_px
        points = np.random.rand(num_points, 2) * np.array([w, h])
        vor = Voronoi(points)
        image = np.ones((h, w), dtype=np.uint8) * 255
        for simplex in vor.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                p1 = vor.vertices[simplex[0]]
                p2 = vor.vertices[simplex[1]]
                cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 0),
                         thickness=max(1, int(0.002 * w)))
        final_image = np.clip(image, 0, 255)
        if path:
            self._save_image_with_metadata(final_image, path)
        return final_image

    @staticmethod
    def _create_single_speckle(diameter_px, blur_sigma):
        size = int(diameter_px * 2)
        if size % 2 == 0:
            size += 1
        center = size // 2
        y, x = np.ogrid[-center:center + 1, -center:center + 1]
        mask = x * x + y * y <= (diameter_px / 2.0) ** 2
        speckle_canvas = np.zeros((size, size), dtype=np.float32)
        speckle_canvas[mask] = 1.0
        return cv2.GaussianBlur(speckle_canvas, (0, 0), blur_sigma) if blur_sigma > 0 else speckle_canvas

class OptimalPatternDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimal Pattern Parameters")
        layout = QFormLayout(self)
        self.width_entry = QLineEdit("1920")
        self.height_entry = QLineEdit("1080")
        self.subset_entry = QLineEdit("31")
        layout.addRow("Target Image Width (px):", self.width_entry)
        layout.addRow("Target Image Height (px):", self.height_entry)
        layout.addRow("Target DIC Subset Size (px):", self.subset_entry)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_values(self):
        return int(self.width_entry.text()), int(self.height_entry.text()), int(self.subset_entry.text())

class PatternGeneratorTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.generated_pattern = None
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Left Panel: Controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)

        # Generation Method Group
        method_group = QGroupBox("Pattern Generation Method")
        method_layout = QFormLayout(method_group)
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Standard Speckle (Spray Simulation)",
            "Fractal-like (DLA Simulation)",
            "3D Print / Decal Simulation",
            "Lithography Simulation (Binary)",
            "Legacy Speckle (mm/DPI)",
            "Legacy Voronoi (mm/DPI)"
        ])
        method_layout.addRow("Select Method:", self.method_combo)

        # Parameters Stack
        self.params_stack = QStackedWidget()
        self._create_standard_params()
        self._create_fractal_params()
        self._create_print_params()
        self._create_litho_params()
        self._create_legacy_speckle_params()
        self._create_legacy_voronoi_params()

        # Tools Group
        tools_group = QGroupBox("Tools")
        tools_layout = QVBoxLayout(tools_group)
        self.generate_optimal_btn = QPushButton("Generate Optimal Pattern...")
        self.suggest_params_btn = QPushButton("Suggest Legacy Parameters...")
        tools_layout.addWidget(self.generate_optimal_btn)
        tools_layout.addWidget(self.suggest_params_btn)

        # Actions Group
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        self.generate_btn = QPushButton("Generate Pattern")
        self.save_btn = QPushButton("Save Pattern")
        action_layout.addWidget(self.generate_btn)
        action_layout.addWidget(self.save_btn)

        left_layout.addWidget(method_group)
        left_layout.addWidget(self.params_stack)
        left_layout.addWidget(tools_group)
        left_layout.addWidget(action_group)
        left_layout.addStretch()

        # Right Panel: Viewer and Quality
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        viewer_group = QGroupBox("Pattern Preview")
        viewer_layout = QVBoxLayout(viewer_group)
        self.pattern_label = FitLabel()
        self.pattern_label.setAlignment(Qt.AlignCenter)
        self.pattern_label.setMinimumSize(400, 400)
        viewer_layout.addWidget(self.pattern_label)

        quality_group = QGroupBox("Scientific Quality Assessment")
        quality_layout = QFormLayout(quality_group)
        self.quality_mig_label = QLabel("MIG: N/A")
        self.quality_size_label = QLabel("Mean Size (px): N/A")
        self.quality_density_label = QLabel("Coverage (%): N/A")
        quality_layout.addRow(self.quality_mig_label)
        quality_layout.addRow(self.quality_size_label)
        quality_layout.addRow(self.quality_density_label)

        right_layout.addWidget(viewer_group)
        right_layout.addWidget(quality_group)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([350, 500])
        layout.addWidget(splitter)
        if hasattr(self.main_window, '_splitters') and isinstance(self.main_window._splitters, list):
            self.main_window._splitters.append(splitter)

        # Connections
        self.method_combo.currentIndexChanged.connect(self.params_stack.setCurrentIndex)
        self.generate_btn.clicked.connect(self.generate_pattern)
        self.generate_optimal_btn.clicked.connect(self.open_optimal_dialog)
        self.suggest_params_btn.clicked.connect(self.suggest_legacy_parameters)
        self.save_btn.clicked.connect(self.save_pattern)

    def _create_standard_params(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(0, 10, 0, 10)
        self.std_img_width = QLineEdit("1024")
        self.std_img_height = QLineEdit("1024")
        self.std_mean_diam = QLineEdit("4.0")
        self.std_dev_diam = QLineEdit("1.5")
        self.std_coverage = QLineEdit("50")
        layout.addRow("Image Width (px):", self.std_img_width)
        layout.addRow("Image Height (px):", self.std_img_height)
        layout.addRow("Mean Speckle Diameter (px):", self.std_mean_diam)
        layout.addRow("Speckle Diameter Std Dev:", self.std_dev_diam)
        layout.addRow("Coverage Ratio (%):", self.std_coverage)
        self.params_stack.addWidget(widget)

    def _create_fractal_params(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(0, 10, 0, 10)
        self.frac_img_width = QLineEdit("1024")
        self.frac_img_height = QLineEdit("1024")
        self.frac_num_particles = QLineEdit("5000")
        self.frac_particle_size = QLineEdit("2")
        layout.addRow("Image Width (px):", self.frac_img_width)
        layout.addRow("Image Height (px):", self.frac_img_height)
        layout.addRow("Number of Particles:", self.frac_num_particles)
        layout.addRow("Particle Size (px):", self.frac_particle_size)
        self.params_stack.addWidget(widget)

    def _create_print_params(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(0, 10, 0, 10)
        self.print_img_width = QLineEdit("1024")
        self.print_img_height = QLineEdit("1024")
        self.print_dot_diam = QLineEdit("5")
        self.print_coverage = QLineEdit("40")
        layout.addRow("Image Width (px):", self.print_img_width)
        layout.addRow("Image Height (px):", self.print_img_height)
        layout.addRow("Dot Diameter (px):", self.print_dot_diam)
        layout.addRow("Coverage Ratio (%):", self.print_coverage)
        self.params_stack.addWidget(widget)

    def _create_litho_params(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(0, 10, 0, 10)
        self.litho_img_width = QLineEdit("1024")
        self.litho_img_height = QLineEdit("1024")
        self.litho_feature_size = QLineEdit("3")
        self.litho_density = QLineEdit("50")
        layout.addRow("Image Width (px):", self.litho_img_width)
        layout.addRow("Image Height (px):", self.litho_img_height)
        layout.addRow("Min Feature Size (px):", self.litho_feature_size)
        layout.addRow("Density (%):", self.litho_density)
        self.params_stack.addWidget(widget)

    def _create_legacy_speckle_params(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(0, 10, 0, 10)
        self.leg_width_mm = QLineEdit("100")
        self.leg_height_mm = QLineEdit("100")
        self.leg_dpi = QLineEdit("300")
        self.leg_min_speckle = QLineEdit("0.4")
        self.leg_max_speckle = QLineEdit("0.6")
        self.leg_density = QLineEdit("0.15")
        self.leg_contrast = QLineEdit("1.0")
        self.leg_noise = QLineEdit("0")
        layout.addRow("Width (mm):", self.leg_width_mm)
        layout.addRow("Height (mm):", self.leg_height_mm)
        layout.addRow("DPI:", self.leg_dpi)
        layout.addRow("Min size (mm):", self.leg_min_speckle)
        layout.addRow("Max size (mm):", self.leg_max_speckle)
        layout.addRow("Density (0-1):", self.leg_density)
        layout.addRow("Contrast (0-1):", self.leg_contrast)
        layout.addRow("Noise (0-255):", self.leg_noise)
        self.params_stack.addWidget(widget)

    def _create_legacy_voronoi_params(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(0, 10, 0, 10)
        self.leg_v_width_mm = QLineEdit("100")
        self.leg_v_height_mm = QLineEdit("100")
        self.leg_v_dpi = QLineEdit("300")
        self.leg_v_points = QLineEdit("200")
        layout.addRow("Width (mm):", self.leg_v_width_mm)
        layout.addRow("Height (mm):", self.leg_v_height_mm)
        layout.addRow("DPI:", self.leg_v_dpi)
        layout.addRow("Number of points:", self.leg_v_points)
        self.params_stack.addWidget(widget)

    def generate_pattern(self):
        try:
            method_idx = self.method_combo.currentIndex()
            if method_idx == 0: # Standard
                w, h = int(self.std_img_width.text()), int(self.std_img_height.text())
                mean_d = float(self.std_mean_diam.text())
                std_d = float(self.std_dev_diam.text())
                coverage = float(self.std_coverage.text())
                self.generated_pattern = self._generate_standard_speckle(w, h, mean_d, std_d, coverage)
            elif method_idx == 1: # Fractal
                w, h = int(self.frac_img_width.text()), int(self.frac_img_height.text())
                n_particles = int(self.frac_num_particles.text())
                p_size = int(self.frac_particle_size.text())
                self.generated_pattern = self._generate_fractal_speckle(w, h, n_particles, p_size)
            elif method_idx == 2: # 3D Print
                w, h = int(self.print_img_width.text()), int(self.print_img_height.text())
                dot_d = int(self.print_dot_diam.text())
                coverage = float(self.print_coverage.text())
                self.generated_pattern = self._generate_printed_dots(w, h, dot_d, coverage)
            elif method_idx == 3: # Lithography
                w, h = int(self.litho_img_width.text()), int(self.litho_img_height.text())
                feature_size = int(self.litho_feature_size.text())
                density = float(self.litho_density.text())
                self.generated_pattern = self._generate_litho_pattern(w, h, feature_size, density)
            elif method_idx == 4: # Legacy Speckle
                gen = PatternMaster(safe_float(self.leg_width_mm.text()), safe_float(self.leg_height_mm.text()), safe_int(self.leg_dpi.text()))
                self.generated_pattern = gen.generate_speckle_pattern(None,
                    safe_float(self.leg_min_speckle.text()), safe_float(self.leg_max_speckle.text()),
                    safe_float(self.leg_density.text()), safe_float(self.leg_contrast.text()),
                    safe_float(self.leg_noise.text())).astype(np.uint8)
            elif method_idx == 5: # Legacy Voronoi
                gen = PatternMaster(safe_float(self.leg_v_width_mm.text()), safe_float(self.leg_v_height_mm.text()), safe_int(self.leg_v_dpi.text()))
                self.generated_pattern = gen.generate_voronoi_pattern(None, safe_int(self.leg_v_points.text()))

            self.update_preview()
            self.analyze_pattern_quality()
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid parameter value: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not generate pattern: {e}")

    def _generate_standard_speckle(self, w, h, mean_d, std_d, coverage):
        img = np.full((h, w), 255, dtype=np.uint8)
        total_area = w * h
        target_area = total_area * (coverage / 100.0)
        covered_area = 0
        max_attempts = int(target_area) * 2
        for _ in range(max_attempts):
            if covered_area >= target_area: break
            d = max(1, int(np.random.normal(mean_d, std_d)))
            r = d // 2
            x, y = random.randint(r, w-r-1), random.randint(r, h-r-1)
            cv2.circle(img, (x, y), r, (0), -1, cv2.LINE_AA)
            covered_area += np.pi * (r**2)
        return cv2.GaussianBlur(img, (5, 5), 0)

    def _generate_fractal_speckle(self, w, h, n_particles, particle_size):
        img = np.full((h, w), 255, dtype=np.uint8)
        x_center, y_center = w // 2, h // 2
        grid_size = particle_size * 4
        grid = [[] for _ in range((w // grid_size + 1) * (h // grid_size + 1))]
        def to_grid_idx(x, y): return (y // grid_size) * (w // grid_size + 1) + (x // grid_size)
        cv2.circle(img, (x_center, y_center), particle_size, (0), -1, cv2.LINE_AA)
        grid[to_grid_idx(x_center, y_center)].append((x_center, y_center))
        particles_placed = 1
        progress = QProgressDialog("Generating Fractal Pattern...", "Cancel", 0, n_particles, self)
        progress.setWindowModality(Qt.WindowModal)
        while particles_placed < n_particles:
            progress.setValue(particles_placed)
            if progress.wasCanceled(): break
            angle = random.uniform(0, 2 * np.pi)
            radius = min(w, h) * 0.48
            x, y = int(x_center + radius * np.cos(angle)), int(y_center + radius * np.sin(angle))
            for _ in range(w*2):
                x += random.choice([-1, 0, 1])
                y += random.choice([-1, 0, 1])
                x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
                gx, gy = x // grid_size, y // grid_size
                found_neighbor = False
                for i in range(max(0, gx-1), min(w//grid_size, gx+2)):
                    for j in range(max(0, gy-1), min(h//grid_size, gy+2)):
                        idx = j * (w//grid_size + 1) + i
                        for px, py in grid[idx]:
                            if ((px - x)**2 + (py - y)**2) < (grid_size/2)**2:
                                found_neighbor = True
                                break
                        if found_neighbor: break
                    if found_neighbor: break
                if found_neighbor:
                    cv2.circle(img, (x, y), particle_size, (0), -1, cv2.LINE_AA)
                    grid[to_grid_idx(x,y)].append((x,y))
                    particles_placed += 1
                    break
        progress.setValue(n_particles)
        return cv2.GaussianBlur(img, (3, 3), 0)

    def _generate_printed_dots(self, w, h, dot_d, coverage):
        img = np.full((h, w), 255, dtype=np.uint8)
        dot_area = np.pi * ((dot_d / 2.0)**2)
        if dot_area == 0: return img
        num_dots = int((w * h * (coverage / 100.0)) / dot_area)
        for _ in range(num_dots):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            cv2.circle(img, (x, y), dot_d // 2, (0), -1, cv2.LINE_AA)
        return img

    def _generate_litho_pattern(self, w, h, feature_size, density):
        small_w, small_h = w // feature_size, h // feature_size
        rand_arr = np.random.rand(small_h, small_w) < (density / 100.0)
        pattern = np.full((small_h, small_w), 255, dtype=np.uint8)
        pattern[rand_arr] = 0
        return cv2.resize(pattern, (w, h), interpolation=cv2.INTER_NEAREST)

    def open_optimal_dialog(self):
        dialog = OptimalPatternDialog(self)
        if dialog.exec():
            w, h, subset_size = dialog.get_values()
            target_diam = max(3.0, min(5.0, subset_size / 5.0))
            std_dev = target_diam / 4.0
            coverage = 50.0
            self.std_img_width.setText(str(w))
            self.std_img_height.setText(str(h))
            self.std_mean_diam.setText(f"{target_diam:.1f}")
            self.std_dev_diam.setText(f"{std_dev:.1f}")
            self.std_coverage.setText(str(coverage))
            self.method_combo.setCurrentIndex(0)
            self.generate_pattern()
            QMessageBox.information(self, "Optimal Pattern Generated",
                f"Generated a pattern with target speckle size of {target_diam:.1f} px for a subset of {subset_size} px.")

    def suggest_legacy_parameters(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose Sample Image for Suggestion", "", "Image Files (*.png *.jpg *.jpeg *.tif)")
        if not path: return
        try:
            img = cv2.imread(path)
            h, w, _ = img.shape
            dpi = safe_int(self.leg_dpi.text(), 300)
            target_px_size = max(5.0, w / 200.0)
            speckle_size_mm = (target_px_size / float(dpi)) * 25.4
            self.method_combo.setCurrentIndex(4) # Switch to legacy speckle
            QApplication.processEvents()
            self.leg_min_speckle.setText(f"{speckle_size_mm * 0.8:.2f}")
            self.leg_max_speckle.setText(f"{speckle_size_mm * 1.2:.2f}")
            self.leg_density.setText("0.20")
            QMessageBox.information(self, "Suggestion", "Legacy parameters suggested based on image resolution.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze image: {e}")

    def update_preview(self):
        if self.generated_pattern is not None:
            pil_img = Image.fromarray(self.generated_pattern)
            pixmap = pil_to_qpixmap(pil_img)
            self.pattern_label.setPixmap(pixmap)

    def analyze_pattern_quality(self):
        if self.generated_pattern is None: return
        img_gray = self.generated_pattern
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        mig = np.mean(grad_mag)
        self.quality_mig_label.setText(f"MIG: {mig:.2f}")
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            diameters = 2 * np.sqrt(areas / np.pi)
            mean_size = np.mean(diameters)
            total_speckle_area = np.sum(areas)
            coverage = (total_speckle_area / img_gray.size) * 100
            self.quality_size_label.setText(f"Mean Size (px): {mean_size:.2f}")
            self.quality_density_label.setText(f"Coverage (%): {coverage:.2f}")
        else:
            self.quality_size_label.setText("Mean Size (px): N/A")
            self.quality_density_label.setText("Coverage (%): 0.00")

    def save_pattern(self):
        if self.generated_pattern is None:
            QMessageBox.warning(self, "Warning", "No pattern has been generated yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Pattern", "", "PNG Files (*.png);;TIFF Files (*.tif);;JPEG Files (*.jpg)")
        if path:
            try:
                cv2.imwrite(path, self.generated_pattern)
                QMessageBox.information(self, "Success", f"Pattern saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save pattern: {e}")

# -------------------- Camera Calibration Tab --------------------
# =============================================================================
# TAB 6: Camera Calibration Tab (FIXED: No popup on startup)
# =============================================================================
class CameraCalibrationTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.image_paths = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.worker_thread = None
        self.worker = None

        self.paper_sizes = {
            "A0": (841, 1189), "A1": (594, 841), "A2": (420, 594), "A3": (297, 420),
            "A4": (210, 297), "A5": (148, 210), "A6": (105, 148),
            "B0": (1000, 1414), "B1": (707, 1000), "B2": (500, 707), "B3": (353, 500),
            "B4": (250, 353), "B5": (176, 250), "B6": (125, 176),
            "Letter": (215.9, 279.4), "Legal": (215.9, 355.6), "Tabloid": (279.4, 431.8)
        }
        self._init_ui()

    def _init_ui(self):
        from PySide6.QtGui import QIntValidator, QDoubleValidator

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_grid_layout = QGridLayout(left_widget)

        # ... (All widget definitions are the same) ...
        generator_group = QGroupBox("0. Generate Checkerboard Pattern")
        generator_form_layout = QFormLayout(generator_group)
        self.paper_size_combo = QComboBox()
        self.paper_size_combo.addItems(self.paper_sizes.keys())
        self.paper_size_combo.setCurrentText("A4")
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Portrait", "Landscape"])
        self.gen_rows_entry = QLineEdit("9")
        self.gen_cols_entry = QLineEdit("14")
        self.gen_square_size_mm_entry = QLineEdit("15")
        self.suggest_values_btn = QPushButton("Suggest Optimal Values")
        self.generate_pattern_btn = QPushButton("Generate & Save Pattern...")
        self.gen_rows_entry.setValidator(QIntValidator(1, 100))
        self.gen_cols_entry.setValidator(QIntValidator(1, 100))
        self.gen_square_size_mm_entry.setValidator(QDoubleValidator(0.1, 1000.0, 2))
        generator_form_layout.addRow("Paper Size:", self.paper_size_combo)
        generator_form_layout.addRow("Orientation:", self.orientation_combo)
        generator_form_layout.addRow("Rows (inner corners):", self.gen_rows_entry)
        generator_form_layout.addRow("Cols (inner corners):", self.gen_cols_entry)
        generator_form_layout.addRow("Square Size (mm):", self.gen_square_size_mm_entry)
        generator_form_layout.addRow(self.suggest_values_btn)
        generator_form_layout.addRow(self.generate_pattern_btn)
        params_group = QGroupBox("2. Parameters")
        params_form_layout = QFormLayout(params_group)
        self.checkerboard_rows_entry = QLineEdit("9")
        self.checkerboard_cols_entry = QLineEdit("14")
        self.square_size_entry = QLineEdit("15.0")
        self.copy_params_btn = QPushButton("Copy Values from Generator")
        self.checkerboard_rows_entry.setValidator(QIntValidator(1, 100))
        self.checkerboard_cols_entry.setValidator(QIntValidator(1, 100))
        self.square_size_entry.setValidator(QDoubleValidator(0.1, 1000.0, 2))
        params_form_layout.addRow("Inner Corners (Rows):", self.checkerboard_rows_entry)
        params_form_layout.addRow("Inner Corners (Cols):", self.checkerboard_cols_entry)
        params_form_layout.addRow("Square Size (e.g., in mm):", self.square_size_entry)
        params_form_layout.addRow(self.copy_params_btn)
        input_group = QGroupBox("1. Input")
        input_v_layout = QVBoxLayout(input_group)
        self.load_calib_images_btn = QPushButton("Load Checkerboard Images")
        self.calib_image_list = QListWidget()
        input_v_layout.addWidget(self.load_calib_images_btn)
        input_v_layout.addWidget(self.calib_image_list)
        action_group = QGroupBox("3. Actions")
        action_v_layout = QVBoxLayout(action_group)
        self.run_calib_btn = QPushButton("Run Calibration")
        self.calib_progress = QProgressBar()
        self.calib_status_label = QLabel("Status: Ready")
        action_v_layout.addWidget(self.run_calib_btn)
        action_v_layout.addWidget(self.calib_progress)
        action_v_layout.addWidget(self.calib_status_label)
        output_group = QGroupBox("4. Output & Verification")
        output_v_layout = QVBoxLayout(output_group)
        self.reprojection_error_label = QLabel("Reprojection Error: N/A")
        io_layout = QHBoxLayout()
        self.save_calib_btn = QPushButton("Save Results")
        self.load_calib_btn = QPushButton("Load Results")
        io_layout.addWidget(self.save_calib_btn)
        io_layout.addWidget(self.load_calib_btn)
        self.apply_folder_btn = QPushButton("Apply Calibration to Folder...")
        self.view_result_btn = QPushButton("View Undistorted vs. Original")
        self.save_calib_btn.setEnabled(False)
        self.apply_folder_btn.setEnabled(False)
        self.view_result_btn.setEnabled(False)
        output_v_layout.addWidget(self.reprojection_error_label)
        output_v_layout.addLayout(io_layout)
        output_v_layout.addWidget(self.apply_folder_btn)
        output_v_layout.addWidget(self.view_result_btn)
        left_grid_layout.addWidget(generator_group, 0, 0, 2, 1)
        left_grid_layout.addWidget(params_group, 2, 0)
        left_grid_layout.addWidget(input_group, 0, 1)
        left_grid_layout.addWidget(action_group, 1, 1)
        left_grid_layout.addWidget(output_group, 2, 1)
        left_grid_layout.setRowStretch(3, 1)
        left_grid_layout.setColumnStretch(0, 1)
        left_grid_layout.setColumnStretch(1, 1)
        right_widget = QGroupBox("Checkerboard Preview")
        right_layout = QVBoxLayout(right_widget)
        self.preview_label = FitLabel()
        right_layout.addWidget(self.preview_label)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([480, 520])
        layout.addWidget(splitter)
        if hasattr(self.main_window, '_splitters') and isinstance(self.main_window._splitters, list):
            self.main_window._splitters.append(splitter)

        self.generate_pattern_btn.clicked.connect(self.generate_checkerboard_pattern)
        self.suggest_values_btn.clicked.connect(self._suggest_optimal_values)
        self.copy_params_btn.clicked.connect(self._copy_generator_params)
        self.load_calib_images_btn.clicked.connect(self.load_images)
        self.calib_image_list.currentItemChanged.connect(self.update_preview)
        self.run_calib_btn.clicked.connect(self.run_calibration)
        self.save_calib_btn.clicked.connect(self._save_calibration_data)
        self.load_calib_btn.clicked.connect(self._load_calibration_data)
        self.apply_folder_btn.clicked.connect(self._apply_calibration_to_folder)
        self.view_result_btn.clicked.connect(self.show_undistorted_preview)
        
        self._suggest_optimal_values(show_message=False)

    @Slot()
    def _suggest_optimal_values(self, show_message=True):
        try:
            paper_name = self.paper_size_combo.currentText()
            w_mm, h_mm = self.paper_sizes[paper_name]
            if self.orientation_combo.currentText() == "Landscape":
                w_mm, h_mm = h_mm, w_mm

            margin = 15.0
            drawable_w = w_mm - 2 * margin
            drawable_h = h_mm - 2 * margin

            if drawable_w < 50 or drawable_h < 50:
                if show_message: QMessageBox.warning(self, "Warning", "Paper size is too small for a reasonable pattern.")
                return

            target_square_size = max(5.0, min(w_mm, h_mm) * 0.06)
            square_size = round(target_square_size)
            if square_size == 0: square_size = 5

            max_cols = int(drawable_w / square_size) - 1
            max_rows = int(drawable_h / square_size) - 1
            
            if max_cols < 2 or max_rows < 2:
                 if show_message: QMessageBox.warning(self, "Calculation Error", "Could not suggest values. The paper size may be too small.")
                 return

            self.gen_rows_entry.setText(str(max_rows))
            self.gen_cols_entry.setText(str(max_cols))
            self.gen_square_size_mm_entry.setText(str(square_size))
            
            if show_message:
                QMessageBox.information(self, "Suggestion", "Optimal values have been suggested for the selected paper size.")
        
        except Exception as e:
            if show_message: QMessageBox.critical(self, "Error", f"Could not suggest values: {e}")

    # ... (The rest of the class methods are unchanged and correct) ...
    @Slot()
    def generate_checkerboard_pattern(self):
        try:
            rows = safe_int(self.gen_rows_entry.text())
            cols = safe_int(self.gen_cols_entry.text())
            square_mm = safe_float(self.gen_square_size_mm_entry.text())
            
            paper = self.paper_size_combo.currentText()
            w_mm, h_mm = self.paper_sizes[paper]
            if self.orientation_combo.currentText() == "Landscape":
                w_mm, h_mm = h_mm, w_mm
            
            DPI = 300
            page_w_px = int((w_mm / 25.4) * DPI)
            page_h_px = int((h_mm / 25.4) * DPI)
            square_px = int((square_mm / 25.4) * DPI)
            if square_px == 0: raise ValueError("Square size is too small to render at 300 DPI.")

            image = np.full((page_h_px, page_w_px), 255, dtype=np.uint8)
            board_w_px = (cols + 1) * square_px
            board_h_px = (rows + 1) * square_px
            if board_w_px > page_w_px or board_h_px > page_h_px:
                QMessageBox.warning(self, "Warning", "Checkerboard is too large for the selected paper size.")
                return
            offset_x = (page_w_px - board_w_px) // 2
            offset_y = (page_h_px - board_h_px) // 2
            for r in range(rows + 1):
                for c in range(cols + 1):
                    if (r + c) % 2 == 0:
                        cv2.rectangle(image, (offset_x + c * square_px, offset_y + r * square_px), 
                                      (offset_x + (c + 1) * square_px, offset_y + (r + 1) * square_px), 0, -1)
            path, _ = QFileDialog.getSaveFileName(self, "Save Pattern", "checkerboard.png", "PNG (*.png)")
            if path:
                Image.fromarray(image).save(path, dpi=(DPI, DPI))
                QMessageBox.information(self, "Success", f"Pattern saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not generate pattern: {e}")

    @Slot()
    def run_calibration(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Input Error", "Please load checkerboard images.")
            return
        try:
            rows = int(self.checkerboard_rows_entry.text())
            cols = int(self.checkerboard_cols_entry.text())
            size = float(self.square_size_entry.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Parameters must be valid numbers.")
            return

        self.run_calib_btn.setEnabled(False)
        self._update_output_buttons_state(False)
        self.calib_status_label.setText("Status: Running...")
        
        self.worker = CalibrationWorker(self.image_paths, (rows, cols), size)
        self.worker.progress.connect(lambda p, m: [self.calib_progress.setValue(p), self.calib_status_label.setText(f"Status: {m}")])
        self.worker.corners_found.connect(self._draw_detected_corners)
        self.worker.finished.connect(self.on_calibration_finished)
        
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    @Slot(str, object)
    def _draw_detected_corners(self, image_path, corners):
        try:
            img = cv2.imread(image_path)
            rows = int(self.checkerboard_rows_entry.text())
            cols = int(self.checkerboard_cols_entry.text())
            cv2.drawChessboardCorners(img, (cols, rows), corners, True)
            
            h, w, ch = img.shape
            q_img = QImage(img.data, w, h, ch * w, QImage.Format_BGR888).copy()
            self.preview_label.setPixmap(QPixmap.fromImage(q_img))
            QApplication.processEvents()
        except Exception as e:
            print(f"Could not draw corners: {e}")

    def on_calibration_finished(self, success, message, matrix, dist_coeffs, error):
        self.run_calib_btn.setEnabled(True)
        self.calib_progress.setValue(0)
        self.calib_status_label.setText(f"Status: {message}")
        if success:
            self.camera_matrix = matrix
            self.dist_coeffs = dist_coeffs
            self.reprojection_error_label.setText(f"Reprojection Error: {error:.4f} pixels")
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)
            self.reprojection_error_label.setText("Reprojection Error: N/A")
        self._update_output_buttons_state(success)

    @Slot()
    def _save_calibration_data(self):
        if self.camera_matrix is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Calibration Data", "", "NumPy Archive (*.npz)")
        if path:
            try:
                np.savez(path, mtx=self.camera_matrix, dist=self.dist_coeffs)
                QMessageBox.information(self, "Success", f"Calibration data saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save data: {e}")
    
    @Slot()
    def _load_calibration_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Calibration Data", "", "NumPy Archive (*.npz)")
        if path:
            try:
                data = np.load(path)
                self.camera_matrix = data['mtx']
                self.dist_coeffs = data['dist']
                self.reprojection_error_label.setText("Reprojection Error: Loaded from file")
                self.calib_status_label.setText("Status: Calibration data loaded successfully.")
                self._update_output_buttons_state(True)
                QMessageBox.information(self, "Success", "Calibration data loaded.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {e}")

    @Slot()
    def _apply_calibration_to_folder(self):
        if self.camera_matrix is None: return
        source_dir = QFileDialog.getExistingDirectory(self, "Select Folder of Images to Undistort")
        if not source_dir: return

        output_dir = os.path.join(source_dir, "undistorted_output")
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        if not image_files:
            QMessageBox.warning(self, "No Images", "No image files found in the selected directory.")
            return

        progress = QProgressDialog("Undistorting images in folder...", "Cancel", 0, len(image_files), self)
        progress.setWindowModality(Qt.WindowModal)

        params = {"img_files": image_files, "source_dir": source_dir, "output_dir": output_dir,
                  "mtx": self.camera_matrix, "dist": self.dist_coeffs}
        
        self.worker = Worker(self._run_batch_undistort_task, params)
        self.worker.progress.connect(lambda p, m: [progress.setValue(p), progress.setLabelText(m)])
        self.worker.finished.connect(lambda: progress.setValue(len(image_files)))
        self.worker.result.connect(lambda res: QMessageBox.information(self, "Success", f"Batch processing complete. Undistorted images are in:\n{output_dir}"))
        self.worker.error.connect(lambda err: QMessageBox.critical(self, "Error", f"Batch processing failed: {err}"))
        
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.start()

    @staticmethod
    def _run_batch_undistort_task(progress_callback, params):
        total = len(params["img_files"])
        for i, filename in enumerate(params["img_files"]):
            progress_callback(i, f"Processing {filename}...")
            img = cv2.imread(os.path.join(params["source_dir"], filename))
            if img is not None:
                undistorted = cv2.undistort(img, params["mtx"], params["dist"], None, params["mtx"])
                cv2.imwrite(os.path.join(params["output_dir"], filename), undistorted)
        progress_callback(total, "Finished.")
        return {"success": True}

    @Slot()
    def _copy_generator_params(self):
        self.checkerboard_rows_entry.setText(self.gen_rows_entry.text())
        self.checkerboard_cols_entry.setText(self.gen_cols_entry.text())
        self.square_size_entry.setText(self.gen_square_size_mm_entry.text())
        QMessageBox.information(self, "Copied", "Generator parameters have been copied.")
    @Slot()
    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Load Checkerboard Images", "", "Image Files (*.png *.jpg *.bmp *.tif)")
        if files:
            self.image_paths = sorted(files, key=natural_key)
            self.calib_image_list.clear()
            self.calib_image_list.addItems([os.path.basename(f) for f in self.image_paths])
            if self.image_paths:
                self.calib_image_list.setCurrentRow(0)
            self._update_output_buttons_state(True)
    @Slot()
    def update_preview(self, current_item, previous_item):
        if current_item:
            path = self.image_paths[self.calib_image_list.row(current_item)]
            self.preview_label.setPixmap(QPixmap(path))
    @Slot()
    def show_undistorted_preview(self):
        if self.camera_matrix is None or self.dist_coeffs is None or not self.image_paths: return
        current_row = self.calib_image_list.currentRow()
        if current_row < 0: current_row = 0
        img = cv2.imread(self.image_paths[current_row])
        undistorted_img = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
        h, w, _ = img.shape
        comparison_img = np.concatenate((img, undistorted_img), axis=1)
        cv2.putText(comparison_img, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison_img, 'Undistorted', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        dialog = QDialog(self)
        dialog.setWindowTitle("Calibration Result: Original vs. Undistorted")
        layout = QVBoxLayout(dialog)
        label = FitLabel()
        h_comp, w_comp, ch = comparison_img.shape
        q_img = QImage(comparison_img.data, w_comp, h_comp, ch * w_comp, QImage.Format_BGR888).copy()
        label.setPixmap(QPixmap.fromImage(q_img))
        layout.addWidget(label)
        dialog.resize(1200, 600)
        dialog.exec()

    def _update_output_buttons_state(self, enabled):
        """Helper to enable/disable all output-related buttons."""
        is_calib_loaded = (self.camera_matrix is not None and self.dist_coeffs is not None)
        self.save_calib_btn.setEnabled(is_calib_loaded)
        self.apply_folder_btn.setEnabled(is_calib_loaded)
        self.view_result_btn.setEnabled(is_calib_loaded and bool(self.image_paths))
# =============================================================================
# VIRTUAL LAB TAB (REWORKED with frame sequence and UQ guide)
# =============================================================================
class ComparisonCanvas(FigureCanvas):
    """A single Matplotlib canvas for displaying a data field."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, data, title):
        self.axes.clear()
        im = self.axes.imshow(data, cmap='jet')
        self.axes.set_title(title)
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        # Add a colorbar
        divider = make_axes_locatable(self.axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.fig.colorbar(im, cax=cax)
        self.fig.tight_layout()
        self.draw()

# =============================================================================
# VIRTUAL LAB TAB (REWORKED with frame sequence and UQ guide)
# =============================================================================
# ... (Keep the ComparisonCanvas class here) ...

# RENAMED and MODIFIED: This class now only handles dataset generation.
class VirtualLabGeneratorTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.ref_image = None
        # REMOVED: UQ-related attributes are gone
        # self.dic_results = None
        # self.ground_truth = None
        # self.dic_results_path = ""
        # self.ground_truth_path = ""
        self.init_ui()

    def init_ui(self):
        # REMOVED: The main vertical splitter is no longer needed.
        # main_splitter = QSplitter(Qt.Vertical)
        layout = QHBoxLayout(self)
        # layout.addWidget(main_splitter)

        # The layout is now a simple horizontal splitter
        top_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(top_splitter)

        # Left Panel: Controls (this part remains mostly the same)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)

        input_group = QGroupBox("1. Input Speckle Pattern")
        input_layout = QVBoxLayout(input_group)
        self.load_image_btn = QPushButton("Load Pattern from File")
        self.use_generated_btn = QPushButton("Use Pattern from Generator Tab")
        self.input_image_label = QLabel("No reference image loaded.")
        self.input_image_label.setWordWrap(True)
        input_layout.addWidget(self.load_image_btn)
        input_layout.addWidget(self.use_generated_btn)
        input_layout.addWidget(self.input_image_label)

        deform_group = QGroupBox("2. Apply Virtual Deformation")
        deform_layout = QFormLayout(deform_group)
        self.deform_mode_combo = QComboBox()
        self.deform_mode_combo.addItems(["Rigid Body Translation", "Uniform Strain", "Crack Opening Displacement", "Lens Distortion"])
        self.params_stack = QStackedWidget()
        self._create_translation_params()
        self._create_strain_params()
        self._create_crack_params()
        self._create_distortion_params()
        deform_layout.addRow("Deformation Type:", self.deform_mode_combo)
        deform_layout.addRow(self.params_stack)
        self.frame_count_spinbox = QSpinBox()
        self.frame_count_spinbox.setRange(2, 1000)
        self.frame_count_spinbox.setValue(2)
        deform_layout.addRow("Frames in sequence:", self.frame_count_spinbox)
        self.generate_btn = QPushButton("Generate & Save Dataset")
        deform_layout.addRow(self.generate_btn)

        left_layout.addWidget(input_group)
        left_layout.addWidget(deform_group)
        left_layout.addStretch()

        # Right Panel: Previews (this part also remains)
        right_widget = QSplitter(Qt.Vertical)
        ref_preview_group = QGroupBox("Reference Image")
        ref_layout = QVBoxLayout(ref_preview_group)
        self.ref_preview = FitLabel("Load an image to begin")
        ref_layout.addWidget(self.ref_preview)

        def_preview_group = QGroupBox("Final Deformed Image Preview")
        def_layout = QVBoxLayout(def_preview_group)
        self.def_preview = FitLabel("Will be generated")
        def_layout.addWidget(self.def_preview)
        right_widget.addWidget(ref_preview_group)
        right_widget.addWidget(def_preview_group)

        top_splitter.addWidget(left_widget)
        top_splitter.addWidget(right_widget)
        top_splitter.setSizes([350, 650])
        
        # REMOVED: The entire bottom panel for UQ is gone.

        # Connections
        self.load_image_btn.clicked.connect(self.load_image)
        self.use_generated_btn.clicked.connect(self.use_generated_pattern)
        self.deform_mode_combo.currentIndexChanged.connect(self.params_stack.setCurrentIndex)
        self.generate_btn.clicked.connect(self.generate_and_save)
        # REMOVED: UQ button connections are gone.

    # ... (Keep all the _create_*_params methods) ...
    def _create_translation_params(self): # ... same as before
        widget = QWidget()
        layout = QFormLayout(widget)
        self.trans_x = QLineEdit("0.5")
        self.trans_y = QLineEdit("-0.25")
        layout.addRow("Total Translate X (pixels):", self.trans_x)
        layout.addRow("Total Translate Y (pixels):", self.trans_y)
        self.params_stack.addWidget(widget)

    def _create_strain_params(self): # ... same as before
        widget = QWidget()
        layout = QFormLayout(widget)
        self.strain_exx = QLineEdit("0.01")
        self.strain_eyy = QLineEdit("-0.005")
        layout.addRow("Total Strain Exx (e.g., 0.01 for 1%):", self.strain_exx)
        layout.addRow("Total Strain Eyy:", self.strain_eyy)
        self.params_stack.addWidget(widget)

    def _create_crack_params(self): # ... same as before
        widget = QWidget()
        layout = QFormLayout(widget)
        self.crack_cod = QLineEdit("5.0")
        label = QLabel("Simulates a vertical crack opening at the image center.")
        label.setWordWrap(True)
        layout.addRow(label)
        layout.addRow("Total Crack Opening Disp. (pixels):", self.crack_cod)
        self.params_stack.addWidget(widget)

    def _create_distortion_params(self): # ... same as before
        widget = QWidget()
        layout = QFormLayout(widget)
        self.dist_k1 = QLineEdit("-0.1")
        self.dist_k2 = QLineEdit("0.01")
        label = QLabel("Simulates radial lens distortion (k1: barrel, k2: pincushion).")
        label.setWordWrap(True)
        layout.addRow(label)
        layout.addRow("Radial k1:", self.dist_k1)
        layout.addRow("Radial k2:", self.dist_k2)
        self.params_stack.addWidget(widget)


    # ... (Keep load_image, use_generated_pattern, _set_ref_image, and generate_and_save methods) ...
    def load_image(self): # ... same as before
        path, _ = QFileDialog.getOpenFileName(self, "Load Speckle Pattern", "", "Image Files (*.png *.jpg *.bmp *.tif)")
        if path:
            self._set_ref_image(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

    def use_generated_pattern(self): # ... same as before
        pattern = self.main_window.tab5.generated_pattern
        if pattern is not None:
            self._set_ref_image(pattern.copy())
        else:
            QMessageBox.warning(self, "Warning", "No pattern has been generated in the 'Pattern Generator' tab yet.")

    def _set_ref_image(self, img): # ... same as before
        if img is None:
            self.input_image_label.setText("Failed to load image.")
            self.ref_image = None
            return
        self.ref_image = img
        self.input_image_label.setText(f"Image loaded ({img.shape[1]}x{img.shape[0]})")
        self.ref_preview.setPixmap(pil_to_qpixmap(Image.fromarray(img)))
        self.def_preview.setText("Will be generated")

    def generate_and_save(self): # ... same as before
        if self.ref_image is None:
            QMessageBox.warning(self, "Input Error", "Please load a reference speckle pattern first.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Virtual Dataset")
        if not output_dir: return

        h, w = self.ref_image.shape
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        mode = self.deform_mode_combo.currentIndex()
        num_frames = self.frame_count_spinbox.value()
        num_steps = num_frames - 1

        try:
            all_true_u, all_true_v = [], []
            progress = QProgressDialog("Generating image sequence...", "Cancel", 0, num_frames, self)
            progress.setWindowModality(Qt.WindowModal)

            cv2.imwrite(os.path.join(output_dir, "00_reference.tif"), self.ref_image)
            progress.setValue(1)

            for i in range(1, num_frames):
                if progress.wasCanceled(): break
                step_fraction = i / num_steps
                u, v = np.zeros_like(x), np.zeros_like(y)
                map_x, map_y = x, y

                if mode == 0:
                    total_tx, total_ty = float(self.trans_x.text()), float(self.trans_y.text())
                    u, v = np.full_like(x, total_tx * step_fraction), np.full_like(y, total_ty * step_fraction)
                    map_x, map_y = x - u, y - v
                elif mode == 1:
                    total_exx, total_eyy = float(self.strain_exx.text()), float(self.strain_eyy.text())
                    x_c, y_c = x - w/2, y - h/2
                    u, v = (total_exx * step_fraction) * x_c, (total_eyy * step_fraction) * y_c
                    map_x, map_y = x - u, y - v
                elif mode == 2:
                    total_cod = float(self.crack_cod.text())
                    crack_pos_x = w / 2
                    u = np.zeros_like(x)
                    u[x > crack_pos_x] = (total_cod * step_fraction) / 2.0
                    u[x < crack_pos_x] = -(total_cod * step_fraction) / 2.0
                    v = np.zeros_like(y)
                    map_x, map_y = x - u, y - v
                elif mode == 3:
                    k1, k2 = float(self.dist_k1.text()), float(self.dist_k2.text())
                    k1_step, k2_step = k1 * step_fraction, k2 * step_fraction
                    cam_matrix = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
                    dist_coeffs = np.array([k1_step, k2_step, 0, 0], dtype=np.float32)
                    map_x, map_y = cv2.initUndistortRectifyMap(cam_matrix, dist_coeffs, None, cam_matrix, (w,h), cv2.CV_32FC1)
                    u, v = map_x - x, map_y - y

                deformed_image = cv2.remap(self.ref_image, map_x, map_y,
                                            interpolation=cv2.INTER_CUBIC,
                                            borderValue=int(np.mean(self.ref_image)))

                cv2.imwrite(os.path.join(output_dir, f"{i:02d}_deformed.tif"), deformed_image)
                all_true_u.append(u.astype(np.float32))
                all_true_v.append(v.astype(np.float32))
                progress.setValue(i + 1)
                if i == num_steps:
                    self.def_preview.setPixmap(pil_to_qpixmap(Image.fromarray(deformed_image)))

            if not progress.wasCanceled():
                np.savez_compressed(os.path.join(output_dir, "ground_truth.npz"),
                                    true_u=np.array(all_true_u),
                                    true_v=np.array(all_true_v))
                QMessageBox.information(self, "Success", f"Virtual dataset saved successfully in:\n{output_dir}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate dataset: {e}")


# =============================================================================
# TAB 7.2: UNCERTAINTY QUANTIFICATION (NEW CLASS)
# =============================================================================
class UQComparisonTab(QWidget):
    """
    A dedicated tab for Uncertainty Quantification.
    Loads DIC results and a ground truth dataset to produce error plots.
    This entire class was refactored from the bottom half of the original VirtualLabTab.
    """
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.dic_results = None
        self.ground_truth = None
        self.dic_results_path = ""
        self.ground_truth_path = ""
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Controls for UQ
        uq_controls_widget = QWidget()
        uq_controls_layout = QHBoxLayout(uq_controls_widget)
        self.load_dic_btn = QPushButton("Load DIC Results (.bin)") # Updated label
        self.dic_path_label = QLabel("Not loaded.")
        self.load_gt_btn = QPushButton("Load Ground Truth (.npz)") # Updated label
        self.gt_path_label = QLabel("Not loaded.")
        self.frame_combo = QComboBox()
        self.compare_btn = QPushButton("Compare Frame")
        uq_controls_layout.addWidget(self.load_dic_btn)
        uq_controls_layout.addWidget(self.dic_path_label, 1)
        uq_controls_layout.addWidget(self.load_gt_btn)
        uq_controls_layout.addWidget(self.gt_path_label, 1)
        uq_controls_layout.addWidget(QLabel("Frame:"))
        uq_controls_layout.addWidget(self.frame_combo)
        uq_controls_layout.addWidget(self.compare_btn)

        # Plotting Area for UQ
        plot_widget = QWidget()
        plot_layout = QGridLayout(plot_widget)
        self.u_calc_canvas = ComparisonCanvas(self)
        self.u_true_canvas = ComparisonCanvas(self)
        self.u_error_canvas = ComparisonCanvas(self)
        self.v_calc_canvas = ComparisonCanvas(self)
        self.v_true_canvas = ComparisonCanvas(self)
        self.v_error_canvas = ComparisonCanvas(self)
        plot_layout.addWidget(QLabel("<b>Calculated U</b>"), 0, 0, Qt.AlignCenter)
        plot_layout.addWidget(QLabel("<b>Ground Truth U</b>"), 0, 1, Qt.AlignCenter)
        plot_layout.addWidget(QLabel("<b>Error U</b>"), 0, 2, Qt.AlignCenter)
        plot_layout.addWidget(self.u_calc_canvas, 1, 0)
        plot_layout.addWidget(self.u_true_canvas, 1, 1)
        plot_layout.addWidget(self.u_error_canvas, 1, 2)
        plot_layout.addWidget(QLabel("<b>Calculated V</b>"), 2, 0, Qt.AlignCenter)
        plot_layout.addWidget(QLabel("<b>Ground Truth V</b>"), 2, 1, Qt.AlignCenter)
        plot_layout.addWidget(QLabel("<b>Error V</b>"), 2, 2, Qt.AlignCenter)
        plot_layout.addWidget(self.v_calc_canvas, 3, 0)
        plot_layout.addWidget(self.v_true_canvas, 3, 1)
        plot_layout.addWidget(self.v_error_canvas, 3, 2)

        layout.addWidget(uq_controls_widget)
        layout.addWidget(plot_widget, 1) # Give plot area more stretch factor

        # Connections
        self.load_dic_btn.clicked.connect(self.load_dic_results)
        self.load_gt_btn.clicked.connect(self.load_ground_truth)
        self.compare_btn.clicked.connect(self.compare_results)

    def load_dic_results(self):
        # This logic is moved from the old VirtualLabTab and updated to load .bin files
        path, _ = QFileDialog.getOpenFileName(self, "Load DIC Result File", "", "Ncorr Binary Files (*.bin)")
        if path:
            try:
                self.dic_results = ncorr.load_DIC_output(path)
                self.dic_results_path = path
                self.dic_path_label.setText(os.path.basename(path))
                if self.ground_truth: # If ground truth is already loaded
                    num_frames = len(self.dic_results.disps)
                    self.frame_combo.clear()
                    self.frame_combo.addItems([f"{i+1}" for i in range(num_frames)])
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load DIC result file: {e}")

    def load_ground_truth(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Ground Truth File", "", "NumPy Archives (*.npz)")
        if path:
            try:
                data = np.load(path)
                if 'true_u' in data and 'true_v' in data:
                    self.ground_truth = data
                    self.ground_truth_path = path
                    self.gt_path_label.setText(os.path.basename(path))
                    num_frames = len(data['true_u'])
                    self.frame_combo.clear()
                    self.frame_combo.addItems([f"{i+1}" for i in range(num_frames)])
                else:
                    QMessageBox.warning(self, "Warning", "Invalid ground truth file. 'true_u' or 'true_v' not found.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load ground truth file: {e}")

    def compare_results(self):
        # This is the updated method from the previous fix
        if self.dic_results is None or self.ground_truth is None:
            QMessageBox.warning(self, "Warning", "Please load both DIC results and ground truth files.")
            return
        try:
            frame_idx = self.frame_combo.currentIndex()
            if frame_idx < 0: return

            disp_frame = self.dic_results.disps[frame_idx]
            from main_app import array2d_to_numpy
            u_calc_grid = array2d_to_numpy(disp_frame.get_u().get_array())
            v_calc_grid = array2d_to_numpy(disp_frame.get_v().get_array())
            
            u_true = self.ground_truth['true_u'][frame_idx]
            v_true = self.ground_truth['true_v'][frame_idx]

            if u_calc_grid.shape != u_true.shape:
                QMessageBox.warning(self, "Shape Mismatch", f"Calculated shape {u_calc_grid.shape} vs ground truth {u_true.shape}.")
            
            u_error = u_calc_grid - u_true
            v_error = v_calc_grid - v_true
            
            self.u_calc_canvas.plot(u_calc_grid, "Calculated U")
            self.u_true_canvas.plot(u_true, "Ground Truth U")
            self.u_error_canvas.plot(u_error, "Error U")
            self.v_calc_canvas.plot(v_calc_grid, "Calculated V")
            self.v_true_canvas.plot(v_true, "Ground Truth V")
            self.v_error_canvas.plot(v_error, "Error V")
        except Exception as e:
            QMessageBox.critical(self, "Comparison Error", f"An error occurred: {e}")

# -------------------- Main Window --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._splitters = []
        self.setWindowTitle("DICStudio - Pre-processing Module (PySide6, Python 3.8) | Creator: ALi Mirjafari")
        g = self.screen().availableGeometry()
        self.setMinimumSize(900, 700)
        self.resize(int(g.width()*0.86), int(g.height()*0.86))
        self.move(g.center() - self.rect().center())
        self.settings = QSettings("DICStudio", "Preprocessing")
        if (b := self.settings.value('preproc/geom')): self.restoreGeometry(b)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tabs
        self.tab1 = FrameExtractorTab(self)
        self.tab2 = ROITab(self)
        self.tab3 = NcorrConfigTab(self)
        self.tab4 = QualityAssessmentTab(self)
        self.tab5 = PatternGeneratorTab(self)
        self.tab6 = CameraCalibrationTab(self)
        self.tab7_1 = VirtualLabGeneratorTab(self) 
        self.tab7_2 = UQComparisonTab(self)     

        self.tabs.addTab(self.tab1, "1. Frame Processor")
        self.tabs.addTab(self.tab2, "2. ROI Configuration")
        self.tabs.addTab(self.tab3, "3. Ncorr DIC Presets")
        self.tabs.addTab(self.tab4, "4. Image Quality")
        self.tabs.addTab(self.tab5, "5. Pattern Generator")
        self.tabs.addTab(self.tab6, "6. Camera Calibration")
        self.tabs.addTab(self.tab7_1, "7.1 Virtual Lab")        
        self.tabs.addTab(self.tab7_2, "7.2 Uncertainty Quantification")


    def get_reference_image_path(self):
        """Provides a path to the first processed image for other tabs to use."""
        if self.tab1 and self.tab1.preview_files and self.tab1.extracted_frames_dir:
            return os.path.join(self.tab1.extracted_frames_dir, self.tab1.preview_files[0])
        return None

    def showEvent(self, e):
        super().showEvent(e)
        if self._splitters:
            states = self.settings.value('preproc/splitters')
            if states:
                try:
                    for sp, st in zip(self._splitters, states):
                        sp.restoreState(st)
                except Exception:
                    pass

    def closeEvent(self, e):
        if hasattr(self, 'settings'):
            self.settings.setValue('preproc/geom', self.saveGeometry())
            if self._splitters:
                self.settings.setValue('preproc/splitters', [sp.saveState() for sp in self._splitters])
        super().closeEvent(e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())