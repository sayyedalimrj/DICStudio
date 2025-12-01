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
import re
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, QPoint, Signal, QSize, QRectF, QPointF
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFontMetrics, QBrush, QPainterPath
from PySide6.QtWidgets import QWidget, QLabel, QSizePolicy, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class FitLabel(QLabel):

    clicked = Signal(QPoint)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._orig_pixmap = None
        self._scaled_pixmap = None

    def mousePressEvent(self, event):
        self.clicked.emit(event.position().toPoint())
        super().mousePressEvent(event)

    def setPixmap(self, pm: QPixmap):
        self._orig_pixmap = pm
        self._scaled_pixmap = None  
        self.update()  

    def set_prescaled_pixmap(self, pm: QPixmap):

        self._scaled_pixmap = pm
        super().setPixmap(pm) # Use the default QLabel drawing for this specific case
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._scaled_pixmap = None
        self.update() 

    def paintEvent(self, event):
        """Paints the scaled pixmap without causing a recursive resize loop."""
        if not self._orig_pixmap:
            # If no pixmap is set, use the default QLabel behavior (e.g., to show text)
            super().paintEvent(event)
            return

        # If a pre-scaled pixmap was set, let the base class handle it
        if self._scaled_pixmap and self.pixmap() is self._scaled_pixmap:
            super().paintEvent(event)
            return

        # Create a new scaled pixmap if needed (e.g., after resize)
        if self._scaled_pixmap is None:
            self._scaled_pixmap = self._orig_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        # Draw the scaled pixmap in the center of the widget
        painter = QPainter(self)
        px = (self.width() - self._scaled_pixmap.width()) // 2
        py = (self.height() - self._scaled_pixmap.height()) // 2
        painter.drawPixmap(px, py, self._scaled_pixmap)


class ResponsiveButton(QPushButton):
    """A button with a dynamic height, suitable for a workflow panel."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._update_min_height()

    def _update_min_height(self):
        fm = QFontMetrics(self.font())
        h = int(fm.height() * 2.2)  # Approx. 2 lines of vertical space
        self.setMinimumHeight(max(28, h))

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type().name == "FontChange":
            self._update_min_height()


# --- HELPER FUNCTIONS ---
def natural_key(s: str):
    """A key for natural sorting of strings containing numbers."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def np_to_qimage(gray: np.ndarray) -> QImage:
    """Converts a 2D NumPy array (grayscale) to a QImage."""
    a = np.asarray(gray)
    if a.ndim != 2:
        raise ValueError("np_to_qimage expects a 2D array.")
    if a.dtype != np.uint8:
        if a.max() <= 1.0:
            a = (np.clip(a, 0, 1) * 255.0).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    h, w = a.shape
    return QImage(a.data, w, h, w, QImage.Format_Grayscale8).copy()


def percentile_minmax(
    arr: np.ndarray, mask: Optional[np.ndarray] = None, p_low=1.0, p_high=99.0
):
    """Calculates percentile-based min/max values for contrast stretching."""
    a = np.asarray(arr, dtype=float)
    if mask is not None:
        a = a[mask.astype(bool)]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    return float(np.percentile(a, p_low)), float(np.percentile(a, p_high))


def array2d_to_numpy(A) -> np.ndarray:
    """Converts a 2D array-like object to a NumPy array."""
    return np.ascontiguousarray(A)


class MplCanvas(FigureCanvas):
    """A Matplotlib canvas widget for embedding plots in a PySide6 application."""

    def __init__(self, width=4, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

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
    layers_changed = Signal(list)  # New signal to notify the layer list

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)  # For displaying the brush cursor

        # --- Status ---
        self.img_np: Optional[np.ndarray] = None
        self.final_mask: Optional[np.ndarray] = None
        self.layers: List[dict] = []
        self.current_mode: Tuple[str, str] = ("rect", "add")  # (shape, action)
        self.selected_layer_idx: Optional[int] = None
        self.active_handle: Optional[str] = None

        # --- For Drawing ---
        self.is_drawing = False
        self.draw_start_pt: Optional[QPointF] = None
        self.current_shape: Optional[QRectF] = None
        self.brush_stroke: List[QPointF] = []
        self.brush_size = 10

        # --- For Pan & Zoom ---
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
            # Creates an initial layer from the input mask
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
        if self.zoom < 1e-9: return QPointF()
        return (p - self.offset) / self.zoom

    # --- Mouse Events ---
    def mousePressEvent(self, ev):
        pos_world = self._view_to_world(ev.position())
        shape, action = self.current_mode

        if ev.button() == Qt.MiddleButton or shape == "pan":
            self.is_panning = True
            self.pan_last_pos = ev.position()
            return

        if ev.button() != Qt.LeftButton: return

        # 1. Check for click on control handles
        if self.selected_layer_idx is not None:
            handles = self._get_handles_for_selected_layer()
            for handle_name, rect in handles.items():
                if rect.contains(ev.position()):
                    self.active_handle = handle_name
                    self.is_drawing = True
                    self.draw_start_pt = pos_world
                    return

        # 2. Check for click on a layer to select it
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer["type"] in ("rect", "ellipse"):
                if layer["geom"].contains(pos_world):
                    self.selected_layer_idx = i
                    self.is_drawing = True  # To enable moving
                    self.draw_start_pt = pos_world
                    self.update()
                    self.layers_changed.emit(self.get_layer_names())
                    return

        # 3. Start drawing a new shape
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
            self.update()  # To show the brush cursor
            return

        # Editing the selected layer
        if self.active_handle:
            self._resize_or_move_layer(pos_world)
        # Moving the selected layer
        elif self.selected_layer_idx is not None and self.layers[self.selected_layer_idx]['type'] in ("rect", "ellipse"):
            delta = pos_world - self.draw_start_pt
            self.layers[self.selected_layer_idx]['geom'].translate(delta)
            self.draw_start_pt = pos_world
        # Drawing a new shape
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

        # If we were editing, update the mask
        if self.active_handle or self.selected_layer_idx is not None:
            self._recomposite_mask()

        # If a new shape was drawn, add it to the layers
        elif self.current_shape and self.current_shape.width() > 1 and self.current_shape.height() > 1:
            self.layers.append({
                "type": shape,
                "mode": action,
                "geom": QRectF(self.current_shape)
            })
            self._recomposite_mask()

        elif shape == "brush" and len(self.brush_stroke) > 1:
            self.layers.append({
                "type": "brush",
                "mode": action,
                "points": list(self.brush_stroke),
                "size": self.brush_size
            })
            self._recomposite_mask()

        self.current_shape = None
        self.brush_stroke = []
        self.active_handle = None
        # self.deselect_layer() # Optional: keep the selection after edit

    def wheelEvent(self, ev):
        if self.img_np is None: return
        factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        mouse_pos_view = ev.position()
        mouse_pos_world = self._view_to_world(mouse_pos_view)

        self.zoom *= factor
        new_offset = mouse_pos_view - mouse_pos_world * self.zoom
        self.offset = new_offset
        self.update()

    # --- Internal Logic & Painting ---
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if self.img_np is None:
            p.end(); return

        p.save()
        p.translate(self.offset)
        p.scale(self.zoom, self.zoom)

        # 1. Draw the background image
        pix = QPixmap.fromImage(np_to_qimage(self.img_np))
        p.drawPixmap(0, 0, pix)

        # 2. Draw the final composited mask (improved method for unified display)
        if self.final_mask is not None and self.final_mask.any():
            p.save()
            # Create a semi-transparent green QColor
            mask_color = QColor(0, 255, 0, 90)
            p.setBrush(QBrush(mask_color))
            p.setPen(Qt.NoPen)  # No border for the filled area

            # Create a QPainterPath from the mask
            path = QPainterPath()
            h, w = self.final_mask.shape
            for y in range(h):
                start_x = -1
                for x in range(w):
                    if self.final_mask.item(y, x) and start_x == -1:
                        start_x = x
                    elif not self.final_mask.item(y, x) and start_x != -1:
                        path.addRect(QRectF(start_x, y, x - start_x, 1))
                        start_x = -1
                if start_x != -1:
                    path.addRect(QRectF(start_x, y, w - start_x, 1))

            p.drawPath(path)
            p.restore()


        # 3. Draw layer outlines
        pen_width = 1.5 / self.zoom
        for i, layer in enumerate(self.layers):
            color = QColor("lime") if layer["mode"] == "add" else QColor("red")

            # Draw selected layer thicker
            current_pen = QPen(color, pen_width * 2.5 if i == self.selected_layer_idx else pen_width)
            p.setPen(current_pen)
            p.setBrush(Qt.NoBrush)

            if layer["type"] in ("rect", "ellipse"):
                geom = layer["geom"]
                if layer["type"] == "rect":
                    p.drawRect(geom)
                else:
                    p.drawEllipse(geom)

        # 4. Draw the shape being currently drawn
        if self.current_shape:
            color = QColor("lime") if self.current_mode[1] == "add" else QColor("red")
            p.setPen(QPen(color, pen_width, Qt.DashLine))
            if self.current_mode[0] == "rect":
                p.drawRect(self.current_shape)
            else:
                p.drawEllipse(self.current_shape)

        # 5. Draw the brush stroke being currently drawn
        if self.brush_stroke:
            color = QColor("lime") if self.current_mode[1] == "add" else QColor("red")
            p.setPen(QPen(color, self.brush_size / self.zoom, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPolyline(self.brush_stroke)

        p.restore()  # Restore scale and translation

        # 6. Draw control handles in View space
        if self.selected_layer_idx is not None and not self.is_panning:
            handles = self._get_handles_for_selected_layer()
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("yellow"))
            for rect in handles.values():
                p.drawRect(rect)

        # 7. Draw the brush cursor
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
                if layer["type"] == "rect":
                    cv2.rectangle(layer_mask, (r.x(), r.y()), (r.right(), r.bottom()), 1, -1)
                else:
                    center = (r.x() + r.width() // 2, r.y() + r.height() // 2)
                    axes = (r.width() // 2, r.height() // 2)
                    cv2.ellipse(layer_mask, center, axes, 0, 0, 360, 1, -1)

            elif layer["type"] == "brush":
                pts = np.array([[int(p.x()), int(p.y())] for p in layer["points"]], np.int32)
                size = int(layer["size"])
                cv2.polylines(layer_mask, [pts], isClosed=False, color=1, thickness=size, lineType=cv2.LINE_AA)

            elif layer["type"] == "imported":
                layer_mask = layer["mask"].astype(np.uint8)

            if layer["mode"] == "add":
                self.final_mask |= (layer_mask > 0)
            else:  # subtract
                self.final_mask &= ~(layer_mask > 0)

        self.update()
        self.mask_changed.emit(self.final_mask.copy())
        self.layers_changed.emit(self.get_layer_names())

    def get_layer_names(self) -> List[str]:
        return [f"Layer {i + 1}: {l['type']} ({l['mode']})" for i, l in enumerate(self.layers)]

    def _get_handles_for_selected_layer(self) -> Dict[str, QRectF]:
        if self.selected_layer_idx is None: return {}
        layer = self.layers[self.selected_layer_idx]
        if layer["type"] not in ("rect", "ellipse"): return {}

        geom = layer["geom"]
        size = 8.0  # Handle size in pixels

        # Convert corner coordinates to View space
        tl = self._world_to_view(geom.topLeft())
        tr = self._world_to_view(geom.topRight())
        bl = self._world_to_view(geom.bottomLeft())
        br = self._world_to_view(geom.bottomRight())

        return {
            "tl": QRectF(tl.x() - size / 2, tl.y() - size / 2, size, size),
            "tr": QRectF(tr.x() - size / 2, tr.y() - size / 2, size, size),
            "bl": QRectF(bl.x() - size / 2, bl.y() - size / 2, size, size),
            "br": QRectF(br.x() - size / 2, br.y() - size / 2, size, size),
        }

    def _resize_or_move_layer(self, pos_world: QPointF):
        if self.selected_layer_idx is None or self.active_handle is None:
            return
            
        layer = self.layers[self.selected_layer_idx]
        geom = layer["geom"]

        # The opposite corner remains fixed based on the active handle
        fixed_corner = QPointF()
        if self.active_handle == "tl": fixed_corner = geom.bottomRight()
        elif self.active_handle == "tr": fixed_corner = geom.bottomLeft()
        elif self.active_handle == "bl": fixed_corner = geom.topRight()
        elif self.active_handle == "br": fixed_corner = geom.topLeft()
        
        if not fixed_corner.isNull():
            layer["geom"] = QRectF(fixed_corner, pos_world).normalized()