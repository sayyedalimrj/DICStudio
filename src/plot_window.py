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
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QComboBox, QCheckBox, QSlider, QLineEdit, QSpinBox, QGroupBox,
    QFileDialog, QMessageBox, QSizePolicy, QPushButton
)
from PySide6.QtCore import Qt, QSettings, QEvent
from PySide6.QtGui import QAction
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
# Assuming a custom MplCanvas based on FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


# Assuming custom_widgets.py and ncorr are available in the project
# A compatible MplCanvas is defined here for completeness, as it's a dependency.
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # We don't use Matplotlib's automatic layout to have more control
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=False)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

def percentile_minmax(a, mask=None):
    """Calculates the 2nd and 98th percentile of an array, ignoring NaNs."""
    if mask is not None:
        a = a[mask]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1.0, 1.0
    return np.percentile(a, 2), np.percentile(a, 98)

def array2d_to_numpy(arr):
    """Dummy conversion function if ncorr types are not standard numpy arrays."""
    return np.array(arr)

# Mock ncorr for standalone execution
class MockROI:
    def get_mask(self): return np.ones((100, 100), dtype=bool)
class MockArray2D:
    def __init__(self, data): self._data = data
    def get_array(self): return self._data
class MockDisp:
    def get_u(self): return MockArray2D(np.random.rand(100, 100))
    def get_v(self): return MockArray2D(np.random.rand(100, 100))
    def get_roi(self): return MockROI()
class MockStrain:
    def get_eyy(self): return MockArray2D(np.random.rand(100, 100))
    def get_exy(self): return MockArray2D(np.random.rand(100, 100))
    def get_exx(self): return MockArray2D(np.random.rand(100, 100))
    def get_roi(self): return MockROI()
class MockDICOut:
    def __init__(self, n): self.disps = [MockDisp() for _ in range(n)]
class MockStrainOut:
    def __init__(self, n): self.strains = [MockStrain() for _ in range(n)]

ncorr = None # This would be the actual imported module in a real scenario


class ViewPlotsWindow(QMainWindow):
    def __init__(self, parent=None, dic_out=None, strain_out=None,
                 ref_image=None, image_paths=None,
                 units: str = "", units_per_pixel: float = 0.0):
        super().__init__(parent)
        self.setWindowTitle("View Plots")
        
        g = self.screen().availableGeometry()
        self.setMinimumSize(900, 600)
        self.resize(int(g.width()*0.86), int(g.height()*0.86))
        self.move(g.center() - self.rect().center())
        self.settings = QSettings()
        if (b := self.settings.value('plots/geom')): self.restoreGeometry(b)

        # Data
        self.dic_out = dic_out or MockDICOut(5)
        self.strain_out = strain_out or MockStrainOut(5)
        self.ref_image = ref_image if ref_image is not None else np.zeros((100, 100))
        self.image_paths = image_paths or []
        self.n_frames = max(len(getattr(self.dic_out, "disps", [])),
                            len(getattr(self.strain_out, "strains", [])))
        self.frame_idx = max(0, self.n_frames - 1)
        self.units = (units or "").strip()
        self.units_per_pixel = float(units_per_pixel or 0.0)

        # UI
        main = QWidget(); self.setCentralWidget(main)
        hbox = QHBoxLayout(main)
        
        self._add_toolbar()

        # Build controls and install event filters
        controls_widget = self._build_controls()
        self.le_lower.installEventFilter(self)
        self.le_upper.installEventFilter(self)
        hbox.addWidget(controls_widget, 1)
        
        # Canvas setup
        self.canvas = MplCanvas(width=7, height=7, dpi=100)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(0, 0)
        self.ax = self.canvas.ax
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Fixed colorbar slot
        self._divider = make_axes_locatable(self.ax)
        self._cax = self._divider.append_axes("right", size="5%", pad=0.1)

        hbox.addWidget(self.canvas, 4)

        # Matplotlib artists
        self._bg_img = None
        self._overlay_img = None
        self._contour = None
        self._norm = Normalize()
        self._cbar = None
        self._last_shape = None
        
        self._scalebar_artists = []
        self._xy_artists = []

        self._refresh_plot(first_time=True)

    def closeEvent(self, e):
        if hasattr(self, 'settings'):
            self.settings.setValue('plots/geom', self.saveGeometry())
        super().closeEvent(e)
        
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.canvas.draw_idle()

    def _add_toolbar(self):
        toolbar = self.addToolBar("Export")
        
        export_png = QAction("Export PNG", self)
        export_png.triggered.connect(self._export_png)
        toolbar.addAction(export_png)

        export_gif = QAction("Export GIF", self)
        export_gif.triggered.connect(self._export_gif)
        toolbar.addAction(export_gif)
        
        export_3d = QAction("3D Strain", self)
        export_3d.triggered.connect(self._open_3d)
        toolbar.addAction(export_3d)

    def _build_controls(self):
        panel = QWidget(); layout = QVBoxLayout(panel)

        # Field
        g0 = QGroupBox("Field"); gl0 = QGridLayout(g0)
        self.cb_type = QComboBox(); self.cb_type.addItems(["U", "V", "EYY", "EXY", "EXX"])
        gl0.addWidget(QLabel("Type:"), 0, 0); gl0.addWidget(self.cb_type, 0, 1)
        layout.addWidget(g0)

        # Frame
        g1 = QGroupBox("Frame"); gl1 = QGridLayout(g1)
        self.slider_frame = QSlider(Qt.Horizontal); self.slider_frame.setRange(0, max(0, self.n_frames-1))
        self.slider_frame.setValue(self.frame_idx)
        self.spn_frame = QSpinBox(); self.spn_frame.setRange(1, max(1, self.n_frames)); self.spn_frame.setValue(self.frame_idx+1)
        gl1.addWidget(QLabel("Index:"), 0, 0); gl1.addWidget(self.slider_frame, 0, 1)
        gl1.addWidget(QLabel("Frame #:"), 1, 0); gl1.addWidget(self.spn_frame, 1, 1)
        layout.addWidget(g1)

        # Local plot options
        g2 = QGroupBox("Local Plot Options"); gl2 = QGridLayout(g2)
        self.sld_alpha = QSlider(Qt.Horizontal); self.sld_alpha.setRange(0, 100); self.sld_alpha.setValue(75)
        
        # --- NEW: Controls for lower/upper bounds with buttons ---
        self.le_lower = QLineEdit()
        self.le_upper = QLineEdit()
        self.le_lower.setPlaceholderText("Lower bound")
        self.le_upper.setPlaceholderText("Upper bound")

        def make_bound_control(line_edit, down_callback, up_callback):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0,0,0,0); layout.setSpacing(2)
            btn_down = QPushButton("-"); btn_down.setFixedWidth(25)
            btn_up = QPushButton("+"); btn_up.setFixedWidth(25)
            layout.addWidget(btn_down)
            layout.addWidget(line_edit)
            layout.addWidget(btn_up)
            btn_down.clicked.connect(down_callback)
            btn_up.clicked.connect(up_callback)
            return widget

        lower_controls = make_bound_control(self.le_lower, 
                                            lambda: self._adjust_bound('lower', 'down'),
                                            lambda: self._adjust_bound('lower', 'up'))
        upper_controls = make_bound_control(self.le_upper, 
                                            lambda: self._adjust_bound('upper', 'down'),
                                            lambda: self._adjust_bound('upper', 'up'))

        gl2.addWidget(QLabel("Transparency:"), 0, 0); gl2.addWidget(self.sld_alpha, 0, 1)
        gl2.addWidget(QLabel("Lower bound:"), 1, 0); gl2.addWidget(lower_controls, 1, 1)
        gl2.addWidget(QLabel("Upper bound:"), 2, 0); gl2.addWidget(upper_controls, 2, 1)
        # --- End of new controls ---
        
        self.spn_contours = QSpinBox(); self.spn_contours.setRange(2, 32); self.spn_contours.setValue(11)
        self.cb_cmap = QComboBox(); self.cb_cmap.addItems(
            ["jet", "viridis", "plasma", "magma", "cividis", "turbo"]
        )
        gl2.addWidget(QLabel("Contours count:"), 3, 0); gl2.addWidget(self.spn_contours, 3, 1)
        gl2.addWidget(QLabel("Colormap:"), 4, 0); gl2.addWidget(self.cb_cmap, 4, 1)
        layout.addWidget(g2)

        # Overlays
        g3 = QGroupBox("Overlays"); gl3 = QGridLayout(g3)
        self.chk_colorbar = QCheckBox("Show colorbar"); self.chk_colorbar.setChecked(True)
        self.chk_axes = QCheckBox("Show axes"); self.chk_axes.setChecked(False)
        self.chk_contour = QCheckBox("Show contour lines"); self.chk_contour.setChecked(False)
        gl3.addWidget(self.chk_colorbar, 0, 0); gl3.addWidget(self.chk_axes, 0, 1); gl3.addWidget(self.chk_contour, 1, 0)
        layout.addWidget(g3)

        # Scale & Guides
        g4 = QGroupBox("Scale & Guides"); gl4 = QGridLayout(g4)
        self.chk_scalebar = QCheckBox("Show scalebar"); self.chk_scalebar.setChecked(True)
        self.le_scalelength = QLineEdit()
        self.le_scalelength.setPlaceholderText("Length")
        self.le_scalelength.setMaximumWidth(100)
        self.le_scalelength.setText("")
        self.chk_xyglyph = QCheckBox("Show XY arrows"); self.chk_xyglyph.setChecked(True)
        self.spn_overlay_fontsize = QSpinBox()
        self.spn_overlay_fontsize.setRange(6, 24); self.spn_overlay_fontsize.setValue(10)

        gl4.addWidget(self.chk_scalebar, 0, 0)
        gl4.addWidget(QLabel("Length (" + (self.units or "px") + "):"), 1, 0)
        gl4.addWidget(self.le_scalelength, 1, 1)
        gl4.addWidget(self.chk_xyglyph, 2, 0)
        gl4.addWidget(QLabel("Overlay Font Size:"), 3, 0)
        gl4.addWidget(self.spn_overlay_fontsize, 3, 1)
        layout.addWidget(g4)

        layout.addStretch()

        # Signals
        self.cb_type.currentIndexChanged.connect(self._refresh_plot)
        self.slider_frame.valueChanged.connect(self._on_frame_slider)
        self.spn_frame.valueChanged.connect(self._on_frame_spin)
        self.sld_alpha.valueChanged.connect(self._refresh_plot)
        self.le_lower.editingFinished.connect(self._refresh_plot)
        self.le_upper.editingFinished.connect(self._refresh_plot)
        self.spn_contours.valueChanged.connect(self._refresh_plot)
        self.cb_cmap.currentIndexChanged.connect(self._refresh_plot)
        self.chk_colorbar.toggled.connect(self._refresh_plot)
        self.chk_axes.toggled.connect(self._refresh_plot)
        self.chk_contour.toggled.connect(self._refresh_plot)
        self.chk_scalebar.toggled.connect(self._refresh_plot)
        self.le_scalelength.editingFinished.connect(self._refresh_plot)
        self.chk_xyglyph.toggled.connect(self._refresh_plot)
        self.spn_overlay_fontsize.valueChanged.connect(self._refresh_plot)

        return panel

    # --- NEW: Event filter for mouse wheel on LineEdits ---
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            if obj is self.le_lower:
                direction = 'down' if event.angleDelta().y() < 0 else 'up'
                self._adjust_bound('lower', direction)
                return True # Event handled, stop further processing
            elif obj is self.le_upper:
                direction = 'down' if event.angleDelta().y() < 0 else 'up'
                self._adjust_bound('upper', direction)
                return True # Event handled

        # Pass other events to the base class
        return super().eventFilter(obj, event)

    # --- NEW: Method to adjust bounds with buttons or wheel ---
    def _adjust_bound(self, which_bound, direction):
        try:
            vmin = float(self.le_lower.text())
            vmax = float(self.le_upper.text())
            
            # Calculate a reasonable step size (2% of the current range)
            step = (vmax - vmin) * 0.02
            if abs(step) < 1e-9: # Handle cases where vmin equals vmax or range is tiny
                step = abs(vmin) * 0.02 if abs(vmin) > 1e-9 else 0.001

            if which_bound == 'lower':
                new_val = vmin - step if direction == 'down' else vmin + step
                self.le_lower.setText(f"{new_val:.6g}")
            else: # upper
                new_val = vmax - step if direction == 'down' else vmax + step
                self.le_upper.setText(f"{new_val:.6g}")

            # Trigger plot refresh
            self._refresh_plot()

        except ValueError:
            return # Ignore if text fields are not valid numbers

    def _on_frame_slider(self, v):
        self.frame_idx = v
        if self.spn_frame.value() != v+1: self.spn_frame.setValue(v+1)
        self._refresh_plot()

    def _on_frame_spin(self, v):
        self.frame_idx = v-1
        if self.slider_frame.value() != self.frame_idx: self.slider_frame.setValue(self.frame_idx)
        self._refresh_plot()

    def _get_field(self, kind: str, idx: int):
        if kind in ("U", "V"):
            disp = self.dic_out.disps[idx]
            data2d = disp.get_u() if kind == "U" else disp.get_v()
            arr = array2d_to_numpy(data2d.get_array())
            mask = array2d_to_numpy(disp.get_roi().get_mask())
        else:
            s = self.strain_out.strains[idx]
            if kind == "EYY": data2d = s.get_eyy()
            elif kind == "EXY": data2d = s.get_exy()
            else: data2d = s.get_exx()
            arr = array2d_to_numpy(data2d.get_array())
            mask = array2d_to_numpy(s.get_roi().get_mask())
        bg = self.ref_image if self.ref_image is not None else np.zeros_like(arr)
        return arr.astype(float), mask.astype(bool), bg

    def _clear_artists(self, lst):
        try:
            for a in lst:
                try: a.remove()
                except Exception: pass
        finally:
            lst.clear()

    def _nice_length(self, target):
        if target <= 0 or not np.isfinite(target): return None
        base = 10.0 ** np.floor(np.log10(target))
        for mul in [1, 2, 5]:
            val = mul * base
            if val >= target * 0.8: return val
        return 10.0 * base

    def _draw_scalebar(self, shape):
        self._clear_artists(self._scalebar_artists)
        if not self.chk_scalebar.isChecked(): return

        H, W = shape
        using_px = (self.units_per_pixel <= 0.0 or not self.units)

        if using_px:
            try: L_px = float(self.le_scalelength.text().strip())
            except Exception: L_px = None
            if not L_px:
                L_px = self._nice_length(0.25 * W) or 100
                self.le_scalelength.setText(str(int(L_px)))
            label, length_px = f"{int(round(L_px))} px", L_px
        else:
            try: L_real = float(self.le_scalelength.text().strip())
            except Exception: L_real = None
            if not L_real:
                target = 0.25 * W * self.units_per_pixel
                L_real = self._nice_length(target) or target
                self.le_scalelength.setText(f"{L_real:.4g}")
            label = f"{L_real:.4g} {self.units}"
            length_px = float(L_real) / float(self.units_per_pixel)

        font_size = self.spn_overlay_fontsize.value()
        fontprops = FontProperties(weight='bold', size=font_size)

        sb = AnchoredSizeBar(self.ax.transData, length_px, label,
                             loc='lower left', pad=0.25, sep=6,
                             borderpad=0.25, frameon=False,
                             size_vertical=max(2, int(0.006 * H)),
                             fontproperties=fontprops, color='white')
        try:
            import matplotlib.patheffects as pe
            sb.txt_label._text.set_path_effects([pe.withStroke(linewidth=2.5, foreground="k")])
        except Exception:
            pass

        self.ax.add_artist(sb)
        self._scalebar_artists.append(sb)

    def _draw_xy_arrows(self, shape):
        self._clear_artists(self._xy_artists)
        if not self.chk_xyglyph.isChecked(): return

        trans = self.ax.transAxes
        origin = (0.04, 0.96); L = 0.08
        font_size = self.spn_overlay_fontsize.value()
        x0, y0 = origin

        kw = dict(arrowprops=dict(arrowstyle='-|>', lw=1.8, color='w'),
                  xycoords=trans, textcoords=trans, zorder=55)
        axX = self.ax.annotate("", xy=(x0 + L, y0), xytext=(x0, y0), **kw)
        axY = self.ax.annotate("", xy=(x0, y0 - L), xytext=(x0, y0), **kw)

        tx = self.ax.text(x0 + L + 0.015, y0, "X", transform=trans, va="center", ha="left",
                          color="w", weight="bold", fontsize=font_size, zorder=56)
        ty = self.ax.text(x0, y0 - L - 0.015, "Y", transform=trans, va="top", ha="center",
                          color="w", weight="bold", fontsize=font_size, zorder=56)
        try:
            import matplotlib.patheffects as pe
            path_effect = [pe.withStroke(linewidth=2.5, foreground="k")]
            for artist in (axX.arrow_patch, axY.arrow_patch, tx, ty):
                artist.set_path_effects(path_effect)
        except Exception:
            pass

        self._xy_artists.extend([axX, axY, tx, ty])

    def _refresh_plot(self, first_time: bool = False):
        if self.n_frames == 0: return
        kind = self.cb_type.currentText()
        idx = max(0, min(self.n_frames - 1, self.frame_idx))
        a, m, bg = self._get_field(kind, idx)
        a[~m] = np.nan

        try: vmin = float(self.le_lower.text())
        except Exception: vmin, _ = percentile_minmax(a, m)
        try: vmax = float(self.le_upper.text())
        except Exception: _, vmax = percentile_minmax(a, m)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = percentile_minmax(a, m)
        
        if not (self.le_lower.hasFocus() or self.le_upper.hasFocus()):
            self.le_lower.setText(f"{vmin:.6g}")
            self.le_upper.setText(f"{vmax:.6g}")

        cmap = get_cmap(self.cb_cmap.currentText())
        alpha = self.sld_alpha.value() / 100.0
        self._norm.vmin, self._norm.vmax = vmin, vmax
        shape = a.shape
        font_size = self.spn_overlay_fontsize.value()

        if first_time or self._bg_img is None or self._last_shape != shape:
            self.ax.clear()
            self._bg_img = self.ax.imshow(bg, cmap='gray', interpolation='nearest', origin='upper')
            self._overlay_img = self.ax.imshow(a, cmap=cmap, norm=self._norm,
                                               interpolation='nearest', origin='upper', alpha=alpha)
            self._last_shape = shape
        else:
            self._bg_img.set_data(bg)
            self._overlay_img.set_data(a)
            self._overlay_img.set_alpha(alpha)
            self._overlay_img.set_cmap(cmap)
            self._overlay_img.set_norm(self._norm)

        self.ax.axis('off' if not self.chk_axes.isChecked() else 'on')

        self._cax.clear()
        if self.chk_colorbar.isChecked():
            self._cax.set_visible(True)
            mappable = ScalarMappable(cmap=cmap, norm=self._norm)
            self._cbar = self.canvas.fig.colorbar(mappable, cax=self._cax, orientation='vertical')
            self._cbar.ax.tick_params(labelsize=font_size)
            
            if kind in ('U', 'V'):
                unit_label = f" ({self.units})" if self.units and self.units != "px" else " (px)"
                label = f"Displacement{unit_label}"
            else:
                label = "Strain"
            self._cbar.set_label(label, size=font_size, weight='bold')
        else:
            self._cax.set_visible(False)

        if self._contour is not None:
            for coll in self._contour.collections: coll.remove()
            self._contour = None
        if self.chk_contour.isChecked():
            n = self.spn_contours.value()
            try:
                levels = np.linspace(vmin, vmax, n)
                self._contour = self.ax.contour(a, levels=levels, colors='k', linewidths=0.6,
                                                alpha=min(1.0, 0.3 + 0.7 * (1 - alpha)))
            except Exception: self._contour = None
        
        self.ax.set_title(f"{kind} — frame {idx + 1}/{self.n_frames}", fontsize=font_size + 2, weight='bold')
        
        self._draw_scalebar(shape)
        self._draw_xy_arrows(shape)

        # This new line optimizes the layout and solves the color bar problem
        self.canvas.fig.tight_layout()
        
        self.canvas.draw()

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "plot.png", "PNG (*.png)")
        if path: self.canvas.fig.savefig(path, dpi=300, bbox_inches="tight")

    def _export_gif(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save GIF", "animation.gif", "GIF (*.gif)")
        if not path: return
        
        try: import imageio.v2 as imageio
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency", "The 'imageio' library is required to export GIFs.")
            return

        kind, cmap = self.cb_type.currentText(), get_cmap(self.cb_cmap.currentText())
        alpha = self.sld_alpha.value() / 100.0
        vmin, vmax = float(self.le_lower.text()), float(self.le_upper.text())
        
        images = []
        for i in range(self.n_frames):
            a, m, bg = self._get_field(kind, i)
            a[~m] = np.nan
            
            fig, ax = plt.subplots(figsize=(self.canvas.fig.get_figwidth(), self.canvas.fig.get_figheight()), dpi=100)
            ax.set_aspect('equal'); ax.axis('off')
            ax.imshow(bg, cmap='gray', interpolation='nearest', origin='upper')
            ax.imshow(a, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, interpolation='nearest', origin='upper')
            ax.set_title(f"{kind} — frame {i + 1}/{self.n_frames}")
            
            if self.chk_colorbar.isChecked():
                mappable = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
                cbar = plt.colorbar(mappable, ax=ax, fraction=0.07, pad=0.04)
                if kind in ('U', 'V'):
                    unit_label = f" ({self.units})" if self.units and self.units != "px" else " (px)"
                    label = f"Displacement{unit_label}"
                else:
                    label = "Strain"
                cbar.set_label(label, weight='bold')


            fig.tight_layout(); fig.canvas.draw()
            images.append(np.asarray(fig.canvas.buffer_rgba()))
            plt.close(fig)

        imageio.mimsave(path, images, fps=10)

    def _open_3d(self):
        kind = self.cb_type.currentText(); idx = self.frame_idx
        a, m, _ = self._get_field(kind, idx)
        a[~m] = np.nan
        Z = np.nan_to_num(a.astype(float))

        step = max(1, int(max(Z.shape) / 150))
        yy, xx = np.mgrid[0:Z.shape[0]:step, 0:Z.shape[1]:step]
        zz = Z[::step, ::step]
        
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        
        cmap = get_cmap(self.cb_cmap.currentText())
        vmin, vmax = float(self.le_lower.text()), float(self.le_upper.text())

        ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cmap, 
                        vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
        
        ax.set_title(f"3D Surface of {kind} (Frame {idx + 1})")
        ax.set_xlabel("X-pixels"); ax.set_ylabel("Y-pixels"); ax.set_zlabel(kind)
        ax.view_init(elev=30, azim=-120)
        fig.tight_layout(); plt.show()