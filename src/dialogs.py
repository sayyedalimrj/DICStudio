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
import os
import json
import numpy as np
import imageio.v2 as iio
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QMessageBox,
    QLabel, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QWidget, QListWidget, QFrame, QFormLayout, QLineEdit,
    QGraphicsScene, QGraphicsView, QDialogButtonBox, QGraphicsRectItem, QSlider,
    QGraphicsPixmapItem, QScrollArea, QListWidgetItem
)
from PySide6.QtCore import Qt, QPoint, QEvent, QPointF, QLineF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QDoubleValidator, QIntValidator, QPixmap, QImage
from matplotlib.patches import Rectangle
from matplotlib import cm

import ncorr
from custom_widgets import ROIDrawer, MplCanvas, np_to_qimage


class FitGraphicsView(QGraphicsView):
    """A QGraphicsView that always fits its content, maintaining aspect ratio."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)


def _enum_names(enum_cls):
    """Helper to get member names from an ncorr enum class."""
    try:
        return [name for name, val in enum_cls.__dict__.items() if isinstance(val, enum_cls)]
    except Exception:
        return []


class CalibrateUnitsDialog(QDialog):
    """
    Two-click calibration: user clicks two points on the reference image,
    then enters the real-world length and unit string.
    """
    def __init__(self, ref_image_np, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate Units")
        self.resize(780, 560)
        self.view = FitGraphicsView(self)
        self.scene = QGraphicsScene(self.view)
        self.view.setScene(self.scene)

        qimg = np_to_qimage(ref_image_np)
        self.pix = QPixmap.fromImage(qimg)
        self.item = self.scene.addPixmap(self.pix)
        self.scene.setSceneRect(QRectF(self.pix.rect()))

        self._points = []
        self._line_item = None
        self._shadow_item = None

        # --- Controls ---
        self.info_label = QLabel("Click two points on the image to define a scale.")
        form = QFormLayout()
        self.ed_units = QLineEdit("mm")
        self.ed_length = QDoubleSpinBox()
        self.ed_length.setDecimals(6); self.ed_length.setMinimum(1e-12)
        self.ed_length.setMaximum(1e12); self.ed_length.setValue(10.0)
        form.addRow("Units:", self.ed_units)
        form.addRow("Real length:", self.ed_length)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)

        # --- Layout ---
        l = QVBoxLayout(self)
        l.addWidget(self.info_label); l.addWidget(self.view, 1)
        l.addLayout(form); l.addWidget(btns)

        self.view.viewport().installEventFilter(self)

        # --- Result attributes ---
        self.units = None
        self.units_per_pixel = None

    def eventFilter(self, obj, ev):
        if obj is self.view.viewport() and ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
            pos = self.view.mapToScene(ev.pos())
            self._points.append(pos)
            if len(self._points) > 2:
                self._points.pop(0)  # Keep only the last two points
            self._update_line()
            return True
        return super().eventFilter(obj, ev)

    def _update_line(self):
        if self._line_item: self.scene.removeItem(self._line_item)
        if self._shadow_item: self.scene.removeItem(self._shadow_item)
        self._line_item = self._shadow_item = None

        if len(self._points) == 1:
            self.info_label.setText("Click a second point.")
        elif len(self._points) == 2:
            self.info_label.setText("Line defined. Adjust length/units and click OK.")
            p1, p2 = self._points
            shadow_pen = QPen(QColor(0, 0, 0, 140), 6.5, Qt.SolidLine, Qt.RoundCap)
            line_pen = QPen(QColor("white"), 3.5, Qt.SolidLine, Qt.RoundCap)
            shadow_pen.setCosmetic(True); line_pen.setCosmetic(True)

            self._shadow_item = self.scene.addLine(QLineF(p1, p2), shadow_pen)
            self._line_item = self.scene.addLine(QLineF(p1, p2), line_pen)
            self._shadow_item.setZValue(1); self._line_item.setZValue(2)

    def _on_accept(self):
        if len(self._points) != 2:
            QMessageBox.warning(self, "Need Two Points", "Please click two points on the image to define the measurement scale.")
            return
        p1, p2 = self._points
        pix_dist = QLineF(p1, p2).length()
        real_len = self.ed_length.value()
        if pix_dist < 1e-6:
            QMessageBox.critical(self, "Invalid Distance", "The pixel distance between the selected points is zero.")
            return
        self.units = self.ed_units.text().strip()
        self.units_per_pixel = real_len / pix_dist
        self.accept()


class SetROIsDialog(QDialog):
    def __init__(self, parent=None, ref_img=None, existing_mask=None):
        super().__init__(parent)
        self.setWindowTitle("Set Region of Interest (ROI)")
        self.setMinimumSize(950, 750)
        self.setModal(True)
        self.roi_mask = None

        # --- FIX: Create roi_drawer INSTANCE FIRST ---
        # This is necessary so that widgets below can connect to its signals.
        self.roi_drawer = ROIDrawer()

        main_layout = QHBoxLayout(self) # Main layout is horizontal

        # --- Left panel: tools and layer list ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(280)

        tools = QGroupBox("Drawing Tools")
        tl = QGridLayout(tools) # Use a grid for better organization
        
        # Tool buttons
        actions = {
            "+ Rect": ('rect', 'add'), "- Rect": ('rect', 'sub'),
            "+ Ellipse": ('ellipse', 'add'), "- Ellipse": ('ellipse', 'sub'),
            "+ Brush": ('brush', 'add'), "- Brush": ('brush', 'sub'),
            "Pan/Zoom": ('pan', None) # Mode is 'pan', action is irrelevant
        }
        row, col = 0, 0
        for name, (shape, mode) in actions.items():
            btn = QPushButton(name)
            # Use a lambda to capture the correct loop variables (shape, mode)
            btn.clicked.connect(lambda checked=False, s=shape, m=mode: self.roi_drawer.set_mode(s, m))
            tl.addWidget(btn, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1
        
        # This needs to go onto the next available grid row.
        row += 1 
        b_clear = QPushButton("Clear All")
        b_clear.clicked.connect(self.roi_drawer.clear_all) 
        tl.addWidget(b_clear, row, 0, 1, 2) # Span across 2 columns
        left_layout.addWidget(tools)

        # Brush tools
        brush_tools = QGroupBox("Brush Tool")
        bl = QFormLayout(brush_tools)
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(2, 100)
        self.brush_slider.setValue(10)
        self.brush_label = QLabel("10 px")
        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        bl.addRow("Size:", self.brush_slider)
        bl.addRow("", self.brush_label)
        left_layout.addWidget(brush_tools)

        # Layer management
        layer_tools = QGroupBox("Layers")
        ll = QVBoxLayout(layer_tools)
        self.layer_list = QListWidget()
        self.layer_list.itemClicked.connect(self._on_layer_selected)
        b_delete_layer = QPushButton("Delete Selected Layer")
        b_delete_layer.clicked.connect(self._delete_selected_layer)
        ll.addWidget(self.layer_list)
        ll.addWidget(b_delete_layer)
        left_layout.addWidget(layer_tools)
        left_layout.addStretch()

        # --- Right panel: ROI drawer and controls ---
        right_panel_layout = QVBoxLayout()
        # The ROIDrawer is already created, now just set its image
        if ref_img is not None:
            self.roi_drawer.set_image(ref_img, existing_mask)
        
        right_panel_layout.addWidget(self.roi_drawer, 1)

        bottom_controls = QHBoxLayout()
        self.lbl_info = QLabel("ROI pixel count: 0")
        b_load = QPushButton("Load Mask from File…")
        b_finish = QPushButton("Finish")
        b_cancel = QPushButton("Cancel")
        b_load.clicked.connect(self._load_roi_file)
        b_finish.clicked.connect(self._finish)
        b_cancel.clicked.connect(self.reject)
        bottom_controls.addWidget(self.lbl_info, 1, Qt.AlignLeft)
        bottom_controls.addWidget(b_load); bottom_controls.addWidget(b_finish); bottom_controls.addWidget(b_cancel)
        right_panel_layout.addLayout(bottom_controls)

        # --- Assemble main layout ---
        main_layout.addWidget(left_panel)
        main_layout.addLayout(right_panel_layout, 1) # Make right panel larger

        # --- Final setup and signal connections ---
        # NOTE: ROIDrawer must emit 'mask_changed' and 'layers_changed' signals
        self.roi_drawer.mask_changed.connect(self._on_mask_updated)
        self.roi_drawer.layers_changed.connect(self._update_layer_list)
        if existing_mask is not None:
            self._on_mask_updated(existing_mask)
        
        # NOTE: ROIDrawer must have a get_layer_names() method
        self._update_layer_list(self.roi_drawer.get_layer_names()) 
        self._on_brush_size_changed(self.brush_slider.value()) # Set initial value

    def _on_brush_size_changed(self, value):
        self.brush_label.setText(f"{value} px")
        # NOTE: ROIDrawer must have a set_brush_size(int) method
        self.roi_drawer.set_brush_size(value)

    def _update_layer_list(self, layer_names: list):
        self.layer_list.clear()
        self.layer_list.addItems(layer_names)

    def _delete_selected_layer(self):
        current_row = self.layer_list.currentRow()
        if current_row >= 0:
            # NOTE: ROIDrawer must have a remove_layer(int) method
            self.roi_drawer.remove_layer(current_row)
            
    def _on_layer_selected(self, item: QListWidgetItem):
        idx = self.layer_list.row(item)
        # NOTE: ROIDrawer must have a selected_layer_idx property
        self.roi_drawer.selected_layer_idx = idx
        self.roi_drawer.update()

    def _on_mask_updated(self, m):
        self.lbl_info.setText(f"ROI pixel count: {int(np.sum(m))}")
        self.roi_mask = m # Store the final mask

    def _load_roi_file(self):
        # This method remains unchanged from the original
        p, _ = QFileDialog.getOpenFileName(self, "Select ROI Image", "", "Image (*.png *.tif *.jpg *.bmp)")
        if not p: return
        try:
            roi = iio.imread(p); roi = roi[..., 0] if roi.ndim == 3 else roi
            m = roi > (np.min(roi) + 1)
            if m.shape != self.roi_drawer.img_np.shape:
                QMessageBox.critical(self, "Mismatch", f"ROI shape {m.shape} must equal image shape {self.roi_drawer.img_np.shape}")
                return
            # This assumes set_image can handle adding a mask as a new layer
            self.roi_drawer.set_image(self.roi_drawer.img_np, existing_mask=m) 
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _finish(self):
        if self.roi_mask is None or not self.roi_mask.any():
            QMessageBox.warning(self, "Empty ROI", "Please define an ROI before finishing.")
            return
        # self.roi_mask has already been updated in _on_mask_updated
        self.accept()
        
    def get_roi_mask(self):
        return self.roi_mask


class SetDICParamsDialog(QDialog):
    """Complete DIC parameter dialog with live subset previews and config import."""
    def __init__(self, parent=None, ref_img=None, roi_mask=None, defaults=None):
        super().__init__(parent)
        self.setWindowTitle("Set DIC Parameters")
        self.setMinimumSize(720, 520)
        self.setModal(True)
        self.ref_img, self.roi_mask = ref_img, roi_mask
        defaults = defaults or {}

        ys, xs = np.where(roi_mask)
        cx = int(np.mean(xs)) if xs.size else ref_img.shape[1]//2
        cy = int(np.mean(ys)) if ys.size else ref_img.shape[0]//2
        self.subset_center = QPoint(cx, cy)

        # Main layout for all content that will go inside the scroll area
        main_layout = QHBoxLayout()
        left = QVBoxLayout(); right = QVBoxLayout()
        main_layout.addLayout(left, 1); main_layout.addLayout(right, 2)

        # --- Left Panel Widgets ---
        gb = QGroupBox("Subset Options"); gl = QGridLayout(gb)
        self.sb_radius = QSpinBox(); self.sb_radius.setRange(10, 200); self.sb_radius.setValue(int(defaults.get("radius", 20)))
        self.sb_spacing = QSpinBox(); self.sb_spacing.setRange(0, 50); self.sb_spacing.setValue(int(defaults.get("spacing", 1)))
        gl.addWidget(QLabel("Subset Radius:"), 0, 0); gl.addWidget(self.sb_radius, 0, 1)
        gl.addWidget(QLabel("Subset Spacing:"), 1, 0); gl.addWidget(self.sb_spacing, 1, 1)
        left.addWidget(gb)

        gb2 = QGroupBox("Solver & Threads"); gl2 = QGridLayout(gb2)
        self.sb_threads = QSpinBox(); self.sb_threads.setRange(1, max(1, os.cpu_count() or 1)); self.sb_threads.setValue(int(defaults.get("threads", max(1, os.cpu_count() or 1))))
        self.chk_show_progress = QCheckBox("Show progress"); self.chk_show_progress.setChecked(bool(defaults.get("debug", False)))
        gl2.addWidget(QLabel("Num Threads:"), 0, 0); gl2.addWidget(self.sb_threads, 0, 1)
        gl2.addWidget(self.chk_show_progress, 1, 0, 1, 2)
        left.addWidget(gb2)

        gb3 = QGroupBox("Interpolation & Subregion"); gl3 = QGridLayout(gb3)
        self.cb_interp = QComboBox(); self.cb_interp.addItems(_enum_names(ncorr.INTERP))
        self.cb_interp.setCurrentText(str(defaults.get("interp", "QUINTIC_BSPLINE_PRECOMPUTE")))
        self.cb_subreg = QComboBox(); self.cb_subreg.addItems(_enum_names(ncorr.SUBREGION))
        self.cb_subreg.setCurrentText(str(defaults.get("subregion", "CIRCLE")))
        gl3.addWidget(QLabel("Interpolation:"), 0, 0); gl3.addWidget(self.cb_interp, 0, 1)
        gl3.addWidget(QLabel("Subregion:"), 1, 0); gl3.addWidget(self.cb_subreg, 1, 1)
        left.addWidget(gb3)

        gb4 = QGroupBox("DIC Analysis Config"); gl4 = QGridLayout(gb4)
        self.cb_dicmode = QComboBox(); self.cb_dicmode.addItems(["preset", "manual"])
        self.cb_dicconfig = QComboBox(); self.cb_dicconfig.addItems(_enum_names(ncorr.DIC_analysis_config))
        self.cb_dicconfig.setCurrentText(str(defaults.get("dic_config", "KEEP_MOST_POINTS")))
        self.dsb_cutoff_corrcoef = QDoubleSpinBox(); self.dsb_cutoff_corrcoef.setDecimals(8); self.dsb_cutoff_corrcoef.setSingleStep(1e-6); self.dsb_cutoff_corrcoef.setValue(float(defaults.get("cutoff_corrcoef", 0.0)))
        self.dsb_update_corrcoef = QDoubleSpinBox(); self.dsb_update_corrcoef.setDecimals(8); self.dsb_update_corrcoef.setSingleStep(1e-6); self.dsb_update_corrcoef.setValue(float(defaults.get("update_corrcoef", 0.0)))
        self.dsb_prctile_corrcoef = QDoubleSpinBox(); self.dsb_prctile_corrcoef.setRange(0, 100); self.dsb_prctile_corrcoef.setValue(float(defaults.get("prctile_corrcoef", 95.0)))
        gl4.addWidget(QLabel("Config mode:"), 0, 0); gl4.addWidget(self.cb_dicmode, 0, 1)
        gl4.addWidget(QLabel("Preset:"), 1, 0); gl4.addWidget(self.cb_dicconfig, 1, 1)
        gl4.addWidget(QLabel("cutoff_corrcoef:"), 2, 0); gl4.addWidget(self.dsb_cutoff_corrcoef, 2, 1)
        gl4.addWidget(QLabel("update_corrcoef:"), 3, 0); gl4.addWidget(self.dsb_update_corrcoef, 3, 1)
        gl4.addWidget(QLabel("prctile_corrcoef:"), 4, 0); gl4.addWidget(self.dsb_prctile_corrcoef, 4, 1)
        left.addWidget(gb4)

        gb5 = QGroupBox("Units & Perspective"); gl5 = QGridLayout(gb5)
        self.cb_units = QComboBox(); self.cb_units.addItems(["", "mm", "cm", "m", "in", "px"])
        self.cb_units.setCurrentText(str(defaults.get("units", "")))
        self.dsb_units_per_px = QDoubleSpinBox(); self.dsb_units_per_px.setDecimals(8); self.dsb_units_per_px.setRange(0.0, 1e9); self.dsb_units_per_px.setValue(float(defaults.get("units_per_pixel", 0.0)))
        self.chk_change_persp = QCheckBox("Change to Eulerian"); self.chk_change_persp.setChecked(bool(defaults.get("change_perspective", False)))
        self.btn_calibrate = QPushButton("Calibrate...")
        gl5.addWidget(QLabel("Units:"), 0, 0); gl5.addWidget(self.cb_units, 0, 1)
        gl5.addWidget(QLabel("Units per pixel:"), 1, 0); gl5.addWidget(self.dsb_units_per_px, 1, 1)
        gl5.addWidget(self.btn_calibrate, 1, 2)
        gl5.addWidget(self.chk_change_persp, 2, 0, 1, 2)
        left.addWidget(gb5)
        
        gb6 = QGroupBox("Output Options")
        gl6 = QGridLayout(gb6)
        self.ed_output_dir = QLineEdit("./ncorr_output") # Default value
        self.btn_browse_dir = QPushButton("Browse...")
        gl6.addWidget(QLabel("Output Directory:"), 0, 0)
        gl6.addWidget(self.ed_output_dir, 1, 0)
        gl6.addWidget(self.btn_browse_dir, 1, 1)
        left.addWidget(gb6)

        left.addStretch()
        
        # --- Right Panel Previews ---
        self.canvas_subset_loc = MplCanvas(width=5, height=4, dpi=90)
        self.canvas_zoomed_subset = MplCanvas(width=5, height=4, dpi=90)
        right.addWidget(QLabel("Subset Location (click to move center)")); right.addWidget(self.canvas_subset_loc)
        right.addWidget(QLabel("Zoomed subset")); right.addWidget(self.canvas_zoomed_subset)
        
        # --- SCROLL AREA IMPLEMENTATION ---
        # 1. Create a content widget and set the main layout on it
        content = QWidget()
        content.setLayout(main_layout)

        # 2. Create the scroll area and configure it
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        # --- Buttons ---
        self.btn_import_cfg = QPushButton("Import Configuration...")
        self.btn_finish = QPushButton("Finish")
        self.btn_cancel = QPushButton("Cancel")

        button_box = QDialogButtonBox()
        button_box.addButton(self.btn_import_cfg, QDialogButtonBox.ActionRole)
        button_box.addButton(self.btn_finish, QDialogButtonBox.AcceptRole)
        button_box.addButton(self.btn_cancel, QDialogButtonBox.RejectRole)
        
        # 3. Create the root layout for the dialog
        root = QVBoxLayout(self)
        # 4. Add scroll area to the root layout
        root.addWidget(scroll)
        # 5. Add buttons to the root layout (outside the scroll area)
        root.addWidget(button_box)

        # --- Signals ---
        self.sb_radius.valueChanged.connect(self._update_previews)
        self.sb_spacing.valueChanged.connect(self._update_previews)
        self.canvas_subset_loc.fig.canvas.mpl_connect('button_press_event', self._on_press)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.btn_import_cfg.clicked.connect(self._import_config)
        
        self.cb_dicmode.currentIndexChanged.connect(self._toggle_manual_fields)
        self.btn_calibrate.clicked.connect(self._open_calibration_dialog)
        self.btn_browse_dir.clicked.connect(self._browse_for_output_directory)

        self._toggle_manual_fields()
        self._update_previews()


    def _browse_for_output_directory(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.ed_output_dir.setText(path)

    def _open_calibration_dialog(self):
        if self.ref_img is None:
            QMessageBox.information(self, "Info", "Reference image is required for calibration.")
            return
        dlg = CalibrateUnitsDialog(self.ref_img, self)
        if dlg.exec():
            self.dsb_units_per_px.setValue(dlg.units_per_pixel)
            self.cb_units.setCurrentText(dlg.units)

    def _import_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open JSON Configuration", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f: cfg = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Invalid JSON", f"Failed to load or parse JSON file.\n\n{e}")
            return

        dic = cfg.get("dic", {})
        if "interp" in dic: self.cb_interp.setCurrentText(dic["interp"])
        if "subregion" in dic: self.cb_subreg.setCurrentText(dic["subregion"])
        if "r" in dic: self.sb_radius.setValue(int(dic["r"]))
        if "spacing" in dic: self.sb_spacing.setValue(int(dic["spacing"]))
        if "threads" in dic: self.sb_threads.setValue(int(dic["threads"]))
        if "debug" in dic: self.chk_show_progress.setChecked(bool(dic["debug"]))
        if "units" in dic: self.cb_units.setCurrentText(str(dic["units"]))
        if "units_per_pixel" in dic: self.dsb_units_per_px.setValue(float(dic["units_per_pixel"]))
        if "change_perspective" in dic: self.chk_change_persp.setChecked(bool(dic["change_perspective"]))

        is_manual = any(k in dic for k in ["cutoff_corrcoef", "update_corrcoef", "prctile_corrcoef"])
        if is_manual:
            self.cb_dicmode.setCurrentText("manual")
            if "cutoff_corrcoef" in dic: self.dsb_cutoff_corrcoef.setValue(float(dic["cutoff_corrcoef"]))
            if "update_corrcoef" in dic: self.dsb_update_corrcoef.setValue(float(dic["update_corrcoef"]))
            if "prctile_corrcoef" in dic: self.dsb_prctile_corrcoef.setValue(float(dic["prctile_corrcoef"]))
        elif "config" in dic:
            self.cb_dicmode.setCurrentText("preset")
            self.cb_dicconfig.setCurrentText(dic["config"])

        self._toggle_manual_fields()
        self._update_previews()
        QMessageBox.information(self, "Success", "Configuration loaded.")

    def _toggle_manual_fields(self):
        manual = (self.cb_dicmode.currentText() == "manual")
        for w in (self.dsb_cutoff_corrcoef, self.dsb_update_corrcoef, self.dsb_prctile_corrcoef):
            w.setEnabled(manual)
        self.cb_dicconfig.setEnabled(not manual)

    def _on_press(self, ev):
        if ev.inaxes != self.canvas_subset_loc.ax or ev.xdata is None: return
        x, y = int(ev.xdata), int(ev.ydata)
        if self.roi_mask[y, x]:
            self.subset_center = QPoint(x, y)
            self._update_previews()

    def _update_previews(self):
        r, s = self.sb_radius.value(), self.sb_spacing.value()
        x, y = self.subset_center.x(), self.subset_center.y()
        ax = self.canvas_subset_loc.ax; ax.clear()
        ax.imshow(self.ref_img, cmap='gray'); ax.contour(self.roi_mask, levels=[0.5], colors='g', linewidths=0.8, alpha=0.7)
        rect = Rectangle((x-r, y-r), 2*r+1, 2*r+1, linewidth=1, edgecolor='r', facecolor='none', zorder=10)
        ax.add_patch(rect); ax.set_title(f"Center: ({x}, {y})"); ax.axis('off'); self.canvas_subset_loc.draw()
        az = self.canvas_zoomed_subset.ax; az.clear()
        sub = self.ref_img[max(0,y-r):y+r+1, max(0,x-r):x+r+1]; az.imshow(sub, cmap='gray')
        if s > 0:
            cx, cy = sub.shape[1]//2, sub.shape[0]//2
            for i in range(-r, r+1, s):
                for j in range(-r, r+1, s): az.plot(cx+j, cy+i, 'r+', markersize=5, alpha=0.8)
        az.axis('off'); self.canvas_zoomed_subset.draw()

    def get_params(self):
        mode = self.cb_dicmode.currentText()
        params = {
            'radius': int(self.sb_radius.value()), 'spacing': int(self.sb_spacing.value()),
            'threads': int(self.sb_threads.value()), 'interp': self.cb_interp.currentText(),
            'subregion': self.cb_subreg.currentText(), 'debug': bool(self.chk_show_progress.isChecked()),
            'units': self.cb_units.currentText(), 'units_per_pixel': float(self.dsb_units_per_px.value()),
            'change_perspective': bool(self.chk_change_persp.isChecked()), 'dic_config_mode': mode,
            'output_dir': self.ed_output_dir.text().strip(),
        }
        if mode == "manual":
            params.update({
                'dic_config': 'KEEP_MOST_POINTS', # Placeholder
                'cutoff_corrcoef': float(self.dsb_cutoff_corrcoef.value()),
                'update_corrcoef': float(self.dsb_update_corrcoef.value()),
                'prctile_corrcoef': float(self.dsb_prctile_corrcoef.value()),
            })
        else: # preset
            params['dic_config'] = self.cb_dicconfig.currentText()
        return params


class SetSeedsDialog(QDialog):
    def __init__(self, parent=None, ref_img=None, roi_mask=None, num_threads=4):
        super().__init__(parent)
        self.setWindowTitle("Set Seeds")
        self.setMinimumSize(700, 500); self.setModal(True)
        self.ref_img, self.roi_mask, self.num_threads = ref_img, roi_mask, num_threads
        self.seeds = []

        main = QVBoxLayout(self)
        top = QHBoxLayout(); side = QVBoxLayout()
        self.info = QLabel(f"Click on ROI to place seed 1 of {self.num_threads}")
        self.lst = QListWidget(); btn_clear = QPushButton("Clear")
        side.addWidget(self.info); side.addWidget(self.lst); side.addWidget(btn_clear)
        
        self.canvas = FitGraphicsView(self)
        self.scene = QGraphicsScene(self.canvas)
        self.canvas.setScene(self.scene)
        self.pixmap_item = None
        self.roi_overlay_item = None
        self.seed_items = []

        top.addLayout(side); top.addWidget(self.canvas, 1)
        main.addLayout(top, 1)

        row = QHBoxLayout(); self.btn_finish = QPushButton("Finish"); self.btn_finish.setEnabled(False)
        btn_cancel = QPushButton("Cancel")
        row.addStretch(); row.addWidget(self.btn_finish); row.addWidget(btn_cancel)
        main.addLayout(row)

        self.canvas.viewport().installEventFilter(self)
        btn_clear.clicked.connect(self._clear)
        self.btn_finish.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        self._plot()

    def eventFilter(self, obj, ev):
        if obj is self.canvas.viewport() and ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
            self._on_press(ev)
            return True
        return super().eventFilter(obj, ev)

    def _on_press(self, ev):
        if len(self.seeds) >= self.num_threads: return
        pos = self.canvas.mapToScene(ev.pos())
        x, y = int(pos.x()), int(pos.y())
        h, w = self.ref_img.shape
        if not (0 <= y < h and 0 <= x < w): return
        if not self.roi_mask[y, x]:
            self.info.setText("Seed must be inside ROI.")
            return

        self.seeds.append([x, y]); self.lst.addItem(f"Seed {len(self.seeds)}: ({x},{y})")
        if len(self.seeds) == self.num_threads:
            self.info.setText(f"All {self.num_threads} seeds placed. Click Finish.")
            self.btn_finish.setEnabled(True)
        else:
            self.info.setText(f"Click on ROI to place seed {len(self.seeds) + 1} of {self.num_threads}")
        self._plot()

    def _clear(self):
        self.seeds.clear(); self.lst.clear(); self.btn_finish.setEnabled(False)
        self.info.setText(f"Click on ROI to place seed 1 of {self.num_threads}")
        self._plot()

    def _plot(self):
        for item in self.seed_items: self.scene.removeItem(item)
        self.seed_items.clear()
        
        if self.pixmap_item is None:
            qimg_bg = np_to_qimage(self.ref_img)
            self.pixmap_item = self.scene.addPixmap(QPixmap.fromImage(qimg_bg))
            self.pixmap_item.setZValue(0)
            self.scene.setSceneRect(QRectF(self.pixmap_item.boundingRect()))
            if self.roi_mask is not None:
                h, w = self.roi_mask.shape
                roi_color_img = np.zeros((h, w, 4), dtype=np.uint8)
                roi_color_img[self.roi_mask] = [12, 255, 12, 100]  # Green, semi-transparent
                qimg_roi = QImage(roi_color_img.data, w, h, w * 4, QImage.Format_RGBA8888)
                self.roi_overlay_item = self.scene.addPixmap(QPixmap.fromImage(qimg_roi))
                self.roi_overlay_item.setZValue(1)

        pen = QPen(QColor("red"), 2); radius = 8
        for x, y in self.seeds:
            l1 = self.scene.addLine(x - radius, y, x + radius, y, pen)
            l2 = self.scene.addLine(x, y - radius, x, y + radius, pen)
            l1.setZValue(2); l2.setZValue(2)
            self.seed_items.extend([l1, l2])
        self.canvas.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def get_seeds(self): return self.seeds


class SetStrainParamsDialog(QDialog):
    """Dialog to set strain calculation parameters with live preview."""
    def __init__(self, parent, ref_img, cur_img, dic_meta, default_radius=15):
        super().__init__(parent)
        self.setWindowTitle("Set Strain Parameters")
        self.setModal(True); self.resize(980, 560)

        left = QFrame(self); left.setMinimumWidth(220)
        vl = QVBoxLayout(left); vl.setSpacing(8)

        grp_strain = QGroupBox("Strain Options")
        fl = QFormLayout(grp_strain)
        self.sld_radius = QSlider(Qt.Horizontal); self.sld_radius.setRange(1, 200); self.sld_radius.setValue(default_radius)
        self.ed_radius = QLineEdit(str(default_radius)); self.ed_radius.setValidator(QIntValidator(1, 1000, self))
        fl.addRow("Strain Radius:", self.sld_radius); fl.addRow("", self.ed_radius)

        grp_view = QGroupBox("View Options")
        fl2 = QFormLayout(grp_view)
        self.cmb_persp = QComboBox(); self.cmb_persp.addItems(["Lagrangian", "Eulerian"])
        self.cmb_field = QComboBox(); self.cmb_field.addItems(["U-Displacement", "V-Displacement", "Exx", "Exy", "Eyy"])
        fl2.addRow(self.cmb_persp); fl2.addRow(self.cmb_field)

        grp_disc = QGroupBox("Discontinuous Analysis"); dl = QVBoxLayout(grp_disc)
        self.chk_subset_trunc = QCheckBox("Subset Truncation"); dl.addWidget(self.chk_subset_trunc)

        grp_zoom = QGroupBox("Zoom/Pan"); hb = QHBoxLayout(grp_zoom)
        self.btn_zoom = QPushButton("Zoom"); self.btn_zoom.setCheckable(True); self.btn_zoom.setChecked(True)
        self.btn_pan  = QPushButton("Pan");  self.btn_pan.setCheckable(True)
        hb.addWidget(self.btn_zoom); hb.addWidget(self.btn_pan)

        vl.addWidget(grp_strain); vl.addWidget(grp_view); vl.addWidget(grp_disc)
        vl.addWidget(grp_zoom); vl.addStretch(1)

        self._view = FitGraphicsView(self)
        self._scene = QGraphicsScene(self._view)
        self._view.setScene(self._scene)
        self._pix_bg = None; self._overlay_item = None

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)

        main = QVBoxLayout(self); row = QHBoxLayout()
        row.addWidget(left); row.addWidget(self._view, 1)
        main.addLayout(row, 1); main.addWidget(btns)

        self._ref_np = ref_img; self._cur_np = cur_img
        self._units = dic_meta.get("units", "pixels")
        self._units_per_pixel = float(dic_meta.get("units_per_pixel", 1.0))
        self._field_name = "V"; self._hover = None; self._hover_bg = None
        self._current_perspective_index = -1
        self._view.setMouseTracking(True); self.installEventFilter(self)
        self._load_background(); self._redraw_overlay()

        self.sld_radius.valueChanged.connect(lambda v: self.ed_radius.setText(str(v)))
        self.sld_radius.valueChanged.connect(self._redraw_overlay)
        self.ed_radius.editingFinished.connect(self._sync_radius_from_edit)
        self.cmb_persp.currentIndexChanged.connect(self._redraw_overlay)
        self.cmb_field.currentIndexChanged.connect(self._redraw_overlay)
        self.btn_zoom.toggled.connect(self._toggle_zoom)
        self.btn_pan.toggled.connect(self._toggle_pan)
        
    def eventFilter(self, source, event):
        if source == self and event.type() == QEvent.MouseMove:
            if not self._pix_bg or not self._hover: return True
            p = self._view.mapToScene(self._view.mapFromGlobal(event.globalPosition().toPoint()))
            x = int(p.x()); y = int(p.y())
            if self._pix_bg.boundingRect().contains(p):
                self._hover.setVisible(True); self._hover_bg.setVisible(True)
                self._hover.setPlainText(f"x-pos: {x}\ny-pos: {y}\n{self._field_name}: — {self._units}")
                br = self._hover.boundingRect()
                self._hover.setPos(p.x() + 10, p.y() + 10)
                self._hover_bg.setRect(p.x() + 8, p.y() + 8, br.width()+8, br.height()+8)
            else:
                self._hover.setVisible(False); self._hover_bg.setVisible(False)
        return super().eventFilter(source, event)

    def _install_hover_text(self):
        self._hover_bg = self._scene.addRect(0,0,0,0, QPen(Qt.NoPen), QBrush(QColor(255, 255, 255, 210)))
        self._hover = self._scene.addText("")
        f = self._hover.font(); f.setPointSizeF(9.0); self._hover.setFont(f)
        self._hover.setDefaultTextColor(QColor("#111"))
        self._hover.setZValue(10); self._hover_bg.setZValue(9)
        self._hover.setVisible(False); self._hover_bg.setVisible(False)

    def _load_background(self):
        img_np = self._ref_np if self.cmb_persp.currentIndex() == 0 else self._cur_np
        pm = QPixmap.fromImage(np_to_qimage(img_np))
        
        self._scene.clear()
        
        # Nullify all Python references to items deleted by scene.clear()
        self._pix_bg = None
        self._overlay_item = None
        self._hover = None
        self._hover_bg = None

        self._pix_bg = self._scene.addPixmap(pm)
        self._view.setSceneRect(self._pix_bg.boundingRect())
        self._install_hover_text() # This re-creates hover items
        self._current_perspective_index = self.cmb_persp.currentIndex() # Update state

    def _toggle_zoom(self, on):
        if on: self.btn_pan.setChecked(False)
        self._view.setDragMode(QGraphicsView.NoDrag)

    def _toggle_pan(self, on):
        if on: self.btn_zoom.setChecked(False)
        self._view.setDragMode(QGraphicsView.ScrollHandDrag if on else QGraphicsView.NoDrag)

    def wheelEvent(self, ev):
        if self.btn_zoom.isChecked():
            factor = 1.15 if ev.angleDelta().y() > 0 else 1/1.15
            self._view.scale(factor, factor)
        else: super().wheelEvent(ev)

    def _sync_radius_from_edit(self):
        v = int(self.ed_radius.text()) if self.ed_radius.text().isdigit() else self.sld_radius.value()
        v = max(1, min(200, v))
        self.ed_radius.setText(str(v))
        if v != self.sld_radius.value(): self.sld_radius.setValue(v)
        else: self._redraw_overlay()

    def _redraw_overlay(self):
        # Use the reliable state variable to check if the background needs reloading
        if self.cmb_persp.currentIndex() != self._current_perspective_index:
            self._load_background()
    
        # Now, safely remove the old overlay if it exists
        if self._overlay_item:
            self._scene.removeItem(self._overlay_item)
            self._overlay_item = None
        
        brect = self._pix_bg.boundingRect(); h, w = int(brect.height()), int(brect.width())
        yy, xx = np.linspace(0, 1, h)[:, None], np.linspace(0, 1, w)[None, :]
        field = yy if "V-Displacement" in self.cmb_field.currentText() else xx
        cmap = cm.get_cmap("viridis")
        img = (cmap((field - field.min())/(field.max()-field.min()))*255).astype(np.uint8)
        qimg = QImage(img.data, w, h, 4*w, QImage.Format_RGBA8888)
        
        self._overlay_item = self._scene.addPixmap(QPixmap.fromImage(qimg.copy()))
        self._overlay_item.setOpacity(0.75)
        self._overlay_item.setZValue(2)
        self._pix_bg.setZValue(1)
        self._field_name = self.cmb_field.currentText().split('-')[0]

    def get_params(self):
        if self.exec() != QDialog.Accepted: return None
        return {
            "radius": int(self.ed_radius.text()),
            "perspective": self.cmb_persp.currentIndex(),
            "preview_field": self.cmb_field.currentText(),
            "subset_trunc": self.chk_subset_trunc.isChecked(),
        }