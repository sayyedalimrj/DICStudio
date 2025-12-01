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
from typing import Optional, Tuple, List, Dict, Any, Union
import os
import tempfile
from pathlib import Path
import dataclasses
import json
import io
import contextlib
import csv

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import signal as sp_signal, fft as sp_fft
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import patheffects, style as mstyle, ticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
from matplotlib.cm import get_cmap, ScalarMappable

from PySide6.QtCore import Qt, QRectF, QEvent, QSettings, QPointF, Signal, QByteArray, Signal 
from PySide6.QtGui import QAction, QPixmap, QImage, QPen, QColor, QPainter, QFont, QFontDatabase, QKeySequence, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox, QLabel,
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLineEdit, QGroupBox,
    QComboBox, QCheckBox, QGraphicsScene, QGraphicsView, QToolBar, QTabWidget,
    QDoubleSpinBox, QSlider, QSpinBox, QFormLayout, QScrollArea, QFrame,
    QProgressDialog, QFontComboBox, QRubberBand, QDialog, QMenu, QColorDialog,
    QDialogButtonBox, QSizePolicy, QRadioButton
)

from dialogs import CalibrateUnitsDialog
from custom_widgets import natural_key

try:
    import imageio.v2 as iio
except Exception:
    iio = None

try:
    import ncorr
except Exception as e:
    ncorr = None

try:
    from scipy.interpolate import Akima1DInterpolator
except ImportError:
    Akima1DInterpolator = None


@dataclasses.dataclass
class PlotOptions:
    style: str = 'default'
    dark_mode: bool = False
    color_cycle: str = 'tab10'
    title_text: str = ""
    font_name: str = "Arial"
    font_size: int = 12
    x_scale: str = 'linear'
    y_scale: str = 'linear'
    autoscale_x: bool = True
    autoscale_y: bool = True
    invert_x: bool = False
    invert_y: bool = False
    show_minor_ticks: bool = True
    use_scientific_notation: bool = False
    use_thousands_separator: bool = True
    rotate_x_ticks: int = 0
    show_grid_major: bool = True
    show_grid_minor: bool = False
    grid_alpha: float = 0.5
    grid_linestyle: str = '--'
    show_top_spine: bool = True
    show_right_spine: bool = True
    show_legend: bool = True
    legend_loc: str = 'best'
    legend_draggable: bool = True
    legend_frame: bool = True
    legend_cols: int = 1
    line_width: float = 1.5
    line_style: str = '-'
    marker_style: str = 'None'
    marker_size: int = 5
    marker_fill: bool = True
    show_fill_under: bool = False
    fill_under_alpha: float = 0.2
    show_data_cursor: bool = True
    annotate_peaks: bool = False
    peak_prominence: float = 0.1
    show_stats_box: bool = False
    nan_handling: str = 'Drop'
    outlier_handling: str = 'None'
    outlier_sigma: float = 3.0
    smoothing_type: str = 'None'
    smoothing_window: int = 5
    savgol_polyorder: int = 2
    transform: str = 'None'
    show_derivative: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PlotOptions':
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered_d)

class PlotOptionsDialog(QDialog):
    def __init__(self, options: PlotOptions, parent=None):
        super().__init__(parent)
        self.options = dataclasses.replace(options)
        self.setWindowTitle("Plot Options")
        self.setMinimumWidth(450)

        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.create_general_tab()
        self.create_axes_tab()
        self.create_appearance_tab()
        self.create_processing_tab()

        main_layout.addWidget(self.tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        self.load_options()

    def create_general_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)

        available_styles = ['default', 'classic'] + sorted(mstyle.available)
        self.cb_style = QComboBox(); self.cb_style.addItems(available_styles)
        self.chk_dark_mode = QCheckBox("Dark Mode Theme")

        color_cycles = ['tab10', 'tab20', 'viridis', 'plasma', 'cividis', 'Pastel1', 'Set1']
        self.cb_color_cycle = QComboBox(); self.cb_color_cycle.addItems(color_cycles)

        self.le_title = QLineEdit()
        self.cb_font = QFontComboBox()
        self.sp_fontsize = QSpinBox(); self.sp_fontsize.setRange(6, 48)

        layout.addRow("Style Theme:", self.cb_style)
        layout.addRow(self.chk_dark_mode)
        layout.addRow("Color Cycle:", self.cb_color_cycle)
        layout.addRow("Custom Title:", self.le_title)
        layout.addRow("Font:", self.cb_font)
        layout.addRow("Font Size:", self.sp_fontsize)

        self.tabs.addTab(tab, "General")

    def create_axes_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)

        self.cb_x_scale = QComboBox(); self.cb_x_scale.addItems(['linear', 'log'])
        self.cb_y_scale = QComboBox(); self.cb_y_scale.addItems(['linear', 'log'])
        self.chk_autoscale_x = QCheckBox("Autoscale X")
        self.chk_autoscale_y = QCheckBox("Autoscale Y")
        self.chk_invert_x = QCheckBox("Invert X-axis")
        self.chk_invert_y = QCheckBox("Invert Y-axis")
        self.chk_minor_ticks = QCheckBox("Show Minor Ticks")
        self.sp_rotate_ticks = QSpinBox(); self.sp_rotate_ticks.setRange(0, 90)

        layout.addRow("X-Axis Scale:", self.cb_x_scale)
        layout.addRow("Y-Axis Scale:", self.cb_y_scale)
        layout.addRow(self.chk_autoscale_x, self.chk_autoscale_y)
        layout.addRow(self.chk_invert_x, self.chk_invert_y)
        layout.addRow(self.chk_minor_ticks)
        layout.addRow("Rotate X Ticks:", self.sp_rotate_ticks)

        self.tabs.addTab(tab, "Axes & Ticks")

    def create_appearance_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)

        gb_grid = QGroupBox("Grid & Spines")
        grid_layout = QFormLayout(gb_grid)
        self.chk_grid_major = QCheckBox("Major Grid")
        self.chk_grid_minor = QCheckBox("Minor Grid")
        self.slider_grid_alpha = QSlider(Qt.Horizontal); self.slider_grid_alpha.setRange(0, 100)
        grid_layout.addRow(self.chk_grid_major, self.chk_grid_minor)
        grid_layout.addRow("Grid Alpha:", self.slider_grid_alpha)

        gb_legend = QGroupBox("Legend")
        legend_layout = QFormLayout(gb_legend)
        self.chk_legend = QCheckBox("Show Legend")
        self.cb_legend_loc = QComboBox(); self.cb_legend_loc.addItems(['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'])
        self.chk_legend_draggable = QCheckBox("Draggable")
        legend_layout.addRow(self.chk_legend)
        legend_layout.addRow("Location:", self.cb_legend_loc)
        legend_layout.addRow(self.chk_legend_draggable)

        gb_lines = QGroupBox("Line & Marker")
        line_layout = QFormLayout(gb_lines)
        self.sp_line_width = QDoubleSpinBox(); self.sp_line_width.setRange(0, 10); self.sp_line_width.setSingleStep(0.25)
        self.cb_line_style = QComboBox(); self.cb_line_style.addItems(['-', '--', '-.', ':', 'None'])
        self.cb_marker_style = QComboBox(); self.cb_marker_style.addItems(['None', 'o', 's', '^', 'v', '<', '>', '.', ','])
        self.sp_marker_size = QSpinBox(); self.sp_marker_size.setRange(0, 20)
        self.chk_fill_under = QCheckBox("Fill Under Curve")
        self.slider_fill_alpha = QSlider(Qt.Horizontal); self.slider_fill_alpha.setRange(0, 100)
        line_layout.addRow("Line Width:", self.sp_line_width)
        line_layout.addRow("Line Style:", self.cb_line_style)
        line_layout.addRow("Marker Style:", self.cb_marker_style)
        line_layout.addRow("Marker Size:", self.sp_marker_size)
        line_layout.addRow(self.chk_fill_under)
        line_layout.addRow("Fill Under Alpha:", self.slider_fill_alpha)

        layout.addWidget(gb_grid)
        layout.addWidget(gb_legend)
        layout.addWidget(gb_lines)
        self.tabs.addTab(tab, "Appearance")

    def create_processing_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)

        self.cb_nan = QComboBox(); self.cb_nan.addItems(['Drop', 'FFill', 'BFill', 'Median'])
        self.cb_smooth = QComboBox(); self.cb_smooth.addItems(['None', 'Moving Average', 'Gaussian', 'SavGol'])
        self.sp_smooth_window = QSpinBox(); self.sp_smooth_window.setRange(3, 101); self.sp_smooth_window.setSingleStep(2)
        self.chk_annotate_peaks = QCheckBox("Annotate Peaks")
        self.chk_stats_box = QCheckBox("Show Stats Box")
        self.chk_data_cursor = QCheckBox("Show Data Cursor on Hover")

        layout.addRow("NaN Handling:", self.cb_nan)
        layout.addRow("Smoothing:", self.cb_smooth)
        layout.addRow("Window:", self.sp_smooth_window)
        layout.addRow(self.chk_annotate_peaks)
        layout.addRow(self.chk_stats_box)
        layout.addRow(self.chk_data_cursor)

        self.tabs.addTab(tab, "Processing")

    def load_options(self):
        opts = self.options
        self.cb_style.setCurrentText(opts.style)
        self.chk_dark_mode.setChecked(opts.dark_mode)
        self.cb_color_cycle.setCurrentText(opts.color_cycle)
        self.le_title.setText(opts.title_text)
        self.cb_font.setCurrentFont(QFont(opts.font_name))
        self.sp_fontsize.setValue(opts.font_size)
        self.cb_x_scale.setCurrentText(opts.x_scale)
        self.cb_y_scale.setCurrentText(opts.y_scale)
        self.chk_autoscale_x.setChecked(opts.autoscale_x)
        self.chk_autoscale_y.setChecked(opts.autoscale_y)
        self.chk_grid_major.setChecked(opts.show_grid_major)
        self.slider_grid_alpha.setValue(int(opts.grid_alpha * 100))
        self.chk_legend.setChecked(opts.show_legend)
        self.cb_legend_loc.setCurrentText(opts.legend_loc)
        self.sp_line_width.setValue(opts.line_width)
        self.chk_fill_under.setChecked(opts.show_fill_under)
        self.slider_fill_alpha.setValue(int(opts.fill_under_alpha * 100))
        self.cb_smooth.setCurrentText(opts.smoothing_type)
        self.sp_smooth_window.setValue(opts.smoothing_window)
        self.chk_annotate_peaks.setChecked(opts.annotate_peaks)
        self.chk_data_cursor.setChecked(opts.show_data_cursor)

    def get_options(self) -> PlotOptions:
        opts = self.options
        opts.style = self.cb_style.currentText()
        opts.dark_mode = self.chk_dark_mode.isChecked()
        opts.color_cycle = self.cb_color_cycle.currentText()
        opts.title_text = self.le_title.text()
        opts.font_name = self.cb_font.currentFont().family()
        opts.font_size = self.sp_fontsize.value()
        opts.x_scale = self.cb_x_scale.currentText()
        opts.y_scale = self.cb_y_scale.currentText()
        opts.autoscale_x = self.chk_autoscale_x.isChecked()
        opts.autoscale_y = self.chk_autoscale_y.isChecked()
        opts.show_grid_major = self.chk_grid_major.isChecked()
        opts.grid_alpha = self.slider_grid_alpha.value() / 100.0
        opts.show_legend = self.chk_legend.isChecked()
        opts.legend_loc = self.cb_legend_loc.currentText()
        opts.line_width = self.sp_line_width.value()
        opts.show_fill_under = self.chk_fill_under.isChecked()
        opts.fill_under_alpha = self.slider_fill_alpha.value() / 100.0
        opts.smoothing_type = self.cb_smooth.currentText()
        opts.smoothing_window = self.sp_smooth_window.value()
        opts.annotate_peaks = self.chk_annotate_peaks.isChecked()
        opts.show_data_cursor = self.chk_data_cursor.isChecked()
        return opts

def apply_plot_and_compute(ax: plt.Axes, data_series: Dict[str, np.ndarray], opts: PlotOptions, *, meta: Optional[Dict] = None) -> Dict:
    if meta is None: meta = {}
    computed_results = {}

    processed = {}
    if not data_series or 'x' not in data_series or not any(k != 'x' for k in data_series):
        ax.cla(); ax.text(0.5, 0.5, "No data to display", ha='center', va='center', transform=ax.transAxes); return {}

    x_raw = data_series['x']
    for name, y_raw in data_series.items():
        if name == 'x': continue

        valid_mask = ~np.isnan(x_raw) & ~np.isnan(y_raw)
        x, y = x_raw[valid_mask], y_raw[valid_mask]

        if x.size < 2:
            processed[name] = {'x': x, 'y': y}
            continue

        if opts.smoothing_type != 'None' and x.size > opts.smoothing_window:
            win = opts.smoothing_window
            if opts.smoothing_type == 'Moving Average':
                y = np.convolve(y, np.ones(win)/win, mode='valid')
                x = x[win//2:-((win-1)//2)]
            elif opts.smoothing_type == 'SavGol':
                poly = min(opts.savgol_polyorder, win - 2)
                if win % 2 == 0: win += 1
                if win > len(y): win = len(y) - (1 if len(y) % 2 == 0 else 0)
                if poly >= win: poly = win - 1
                if win > poly and win > 0:
                    y = sp_signal.savgol_filter(y, win, poly)

        processed[name] = {'x': x, 'y': y}

    style_context = mstyle.context(opts.style)
    if opts.dark_mode:
        dark_params = {
            "figure.facecolor": "#333", "axes.facecolor": "#444", "savefig.facecolor": "#333",
            "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white",
            "ytick.color": "white", "axes.edgecolor": "white", "grid.color": "white"
        }
        style_context = contextlib.ExitStack()
        style_context.enter_context(mstyle.context(opts.style))
        style_context.enter_context(plt.rc_context(dark_params))

    with style_context:
        ax.cla()
        ax.set_prop_cycle(color=plt.get_cmap(opts.color_cycle).colors)

        for name, data in processed.items():
            if data['x'].size == 0: continue
            line, = ax.plot(data['x'], data['y'], label=name,
                    linewidth=opts.line_width, linestyle=opts.line_style,
                    marker=opts.marker_style, markersize=opts.marker_size)

            if opts.show_fill_under:
                ax.fill_between(data['x'], data['y'], alpha=opts.fill_under_alpha, color=line.get_color(), edgecolor='none')

            if opts.annotate_peaks and data['y'].size > 3:
                peaks, _ = sp_signal.find_peaks(data['y'], prominence=(np.nanmax(data['y']) - np.nanmin(data['y'])) * opts.peak_prominence)
                ax.plot(data['x'][peaks], data['y'][peaks], "x", c='red', markersize=8)

        if opts.show_legend:
            leg = ax.legend(loc=opts.legend_loc, frameon=opts.legend_frame,
                            ncols=opts.legend_cols)
            if leg: leg.set_draggable(opts.legend_draggable)

        ax.set_xscale(opts.x_scale); ax.set_yscale(opts.y_scale)
        ax.set_autoscale_on(True)
        ax.relim(); ax.autoscale_view()
        if not opts.autoscale_x: ax.set_xlim(ax.get_xlim())
        if not opts.autoscale_y: ax.set_ylim(ax.get_ylim())
        if opts.invert_x: ax.invert_xaxis()
        if opts.invert_y: ax.invert_yaxis()

        title = opts.title_text or meta.get('title', '')
        ax.set_title(title, fontname=opts.font_name, fontsize=opts.font_size)
        ax.set_xlabel(meta.get('xlabel', 'X-axis')); ax.set_ylabel(meta.get('ylabel', 'Y-axis'))

        ax.grid(opts.show_grid_major, which='major', alpha=opts.grid_alpha, ls=opts.grid_linestyle)
        if opts.show_minor_ticks:
            ax.minorticks_on()
            if opts.show_grid_minor:
                 ax.grid(True, which='minor', alpha=opts.grid_alpha * 0.5, ls=':')
        else:
            ax.minorticks_off()

    return computed_results


class MplPopOutWindow(QDialog):
    def __init__(self, fig: Figure, title: str = "Plot Details", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 450)
        layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar); layout.addWidget(self.canvas)

    def update_plot(self):
        self.canvas.draw_idle()

class MplPlotWidget(QWidget):
    plot_updated = Signal()
    options_changed = Signal()

    def __init__(self, width=5, height=3, dpi=100, parent=None):
        super().__init__(parent)
        self.canvas = MplCanvas(width=width, height=height, dpi=dpi)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.btn_pop_out = QPushButton("↗️ Pop-out")
        self.btn_pop_out.setToolTip("Open this plot in a separate, resizable window")
        self.btn_options = QPushButton("Options ▸")
        self.btn_options.setToolTip("Show advanced plot options")

        self.plot_data: Dict[str, np.ndarray] = {}
        self.plot_meta: Dict[str, Any] = {}
        self.options = PlotOptions()
        self.pop_out_window: Optional[MplPopOutWindow] = None
        self.data_cursor_annotation = None

        v_layout = QVBoxLayout(self); v_layout.setContentsMargins(0,0,0,0)
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.toolbar)
        h_layout.addStretch()
        h_layout.addWidget(self.btn_options)
        h_layout.addWidget(self.btn_pop_out)
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.canvas)

        self.btn_pop_out.clicked.connect(self.pop_out)
        self.btn_options.clicked.connect(self.show_options_dialog)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    @property
    def ax(self): return self.canvas.ax

    @property
    def fig(self): return self.canvas.fig

    def set_data(self, data: Dict, meta: Optional[Dict] = None):
        self.plot_data = data if data else {}
        self.plot_meta = meta if meta else {}
        self.update_plot()

    def update_plot(self):
        if not self.plot_data:
            self.ax.cla()
            self.ax.text(0.5, 0.5, "No data available.", ha='center', va='center', transform=self.ax.transAxes)
        else:
            apply_plot_and_compute(self.ax, self.plot_data, self.options, meta=self.plot_meta)

        self.canvas.draw()
        self.plot_updated.emit()

    def show_options_dialog(self):
        dialog = PlotOptionsDialog(self.options, self)
        if dialog.exec():
            self.options = dialog.get_options()
            self.options_changed.emit()
            self.update_plot()

    def on_mouse_move(self, event):
        if not self.options.show_data_cursor or event.inaxes != self.ax:
            if self.data_cursor_annotation and self.data_cursor_annotation.get_visible():
                self.data_cursor_annotation.set_visible(False)
                self.canvas.draw_idle()
            return

        if not self.ax.lines: return

        min_dist = float('inf')
        target_point = None
        target_line = None

        for line in self.ax.lines:
            if not line.get_visible() or not hasattr(line, 'get_xydata'): continue

            xy_data = line.get_xydata()
            if len(xy_data) == 0: continue

            trans_data = self.ax.transData.transform(xy_data)
            mouse_pos = (event.x, event.y)

            distances = np.sqrt(np.sum((trans_data - mouse_pos)**2, axis=1))
            idx = np.argmin(distances)

            if distances[idx] < min_dist:
                min_dist = distances[idx]
                target_point = (xy_data[idx, 0], xy_data[idx, 1])

        if target_point and min_dist < 20:
            if not self.data_cursor_annotation:
                self.data_cursor_annotation = self.ax.annotate("", xy=(0,0), xytext=(15, -15),
                    textcoords="offset points", bbox=dict(boxstyle="round,pad=0.4", fc="orange", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", facecolor='black'))

            self.data_cursor_annotation.xy = target_point
            self.data_cursor_annotation.set_text(f"X: {target_point[0]:.3g}\nY: {target_point[1]:.3g}")
            self.data_cursor_annotation.set_visible(True)
            self.canvas.draw_idle()
        else:
            if self.data_cursor_annotation and self.data_cursor_annotation.get_visible():
                self.data_cursor_annotation.set_visible(False)
                self.canvas.draw_idle()

    def show_context_menu(self, pos):
        menu = QMenu(self)

        def add_toggle_action(text, opt_attr):
            action = QAction(text, self, checkable=True)
            action.setChecked(getattr(self.options, opt_attr))
            action.toggled.connect(lambda checked: (setattr(self.options, opt_attr, checked), self.update_plot()))
            menu.addAction(action)

        add_toggle_action("Show Grid", "show_grid_major")
        add_toggle_action("Show Legend", "show_legend")
        add_toggle_action("Dark Mode", "dark_mode")
        add_toggle_action("Data Cursor", "show_data_cursor")
        menu.addSeparator()
        menu.addAction("Autoscale View", self.ax.autoscale)
        menu.addSeparator()
        menu.addAction("Copy Image", self.copy_image_to_clipboard)
        menu.addAction("Export Data (CSV)...", self.export_processed_csv)

        menu.exec(self.mapToGlobal(pos))

    def copy_image_to_clipboard(self):
        with io.BytesIO() as buf:
            self.fig.savefig(buf, format='png', dpi=150)
            QGuiApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
        QMessageBox.information(self, "Copied", "Plot image copied to clipboard.")

    def export_processed_csv(self):
        if not self.plot_data:
            QMessageBox.warning(self, "No Data", "There is no data to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Processed Data", "", "CSV Files (*.csv)")
        if not path: return

        try:
            x_data = self.plot_data.get('x', np.array([]))
            header = ["x"]
            columns = [x_data]

            for key, y_data in self.plot_data.items():
                if key != 'x':
                    header.append(key)
                    columns.append(y_data)

            max_len = max(len(c) for c in columns)
            data_matrix = np.full((max_len, len(columns)), np.nan)
            for i, col in enumerate(columns):
                data_matrix[:len(col), i] = col

            np.savetxt(path, data_matrix, delimiter=',', header=','.join(header), comments='')
            QMessageBox.information(self, "Success", f"Data exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data:\n{e}")

    def pop_out(self):
        if self.pop_out_window is None or not self.pop_out_window.isVisible():
            title = self.ax.get_title() or "Plot Details"
            self.pop_out_window = MplPopOutWindow(self.fig, title, self)
            self.plot_updated.connect(self.pop_out_window.update_plot)
            self.pop_out_window.finished.connect(self.on_pop_out_closed)
            self.pop_out_window.show()

    def on_pop_out_closed(self):
        if self.pop_out_window:
            try: self.plot_updated.disconnect(self.pop_out_window.update_plot)
            except (TypeError, RuntimeError): pass
        self.pop_out_window = None

def np_to_qimage(gray: np.ndarray) -> QImage:
    a = np.asarray(gray)
    if a.ndim != 2: raise ValueError("np_to_qimage expects a 2D array.")
    if a.dtype != np.uint8:
        if a.max() <= 1.0: a = (np.clip(a, 0, 1) * 255.0).astype(np.uint8)
        else: a = np.clip(a, 0, 255).astype(np.uint8)
    if not a.flags['C_CONTIGUOUS']: a = np.ascontiguousarray(a)
    h, w = a.shape
    return QImage(a.data, w, h, w, QImage.Format_Grayscale8).copy()

def array2d_to_numpy(A) -> np.ndarray: return np.array(A, copy=True)

def bilinear_sample(A: np.ndarray, y: float, x: float) -> float:
    h, w = A.shape[:2]
    if h == 0 or w == 0: return float('nan')
    y = float(np.clip(y, 0, h - 1.0001)); x = float(np.clip(x, 0, w - 1.0001))
    y0 = int(y); x0 = int(x)
    y1 = y0 + 1; x1 = x0 + 1
    wy = y - y0; wx = x - x0
    v00 = A[y0, x0]; v01 = A[y0, x1]; v10 = A[y1, x0]; v11 = A[y1, x1]
    v0 = v00 * (1 - wx) + v01 * wx; v1 = v10 * (1 - wx) + v11 * wx
    return float(v0 * (1 - wy) + v1 * wy)

def nearest_sample(A: np.ndarray, y: float, x: float) -> float:
    y0 = int(round(float(np.clip(y, 0, A.shape[0] - 1))))
    x0 = int(round(float(np.clip(x, 0, A.shape[1] - 1))))
    return float(A[y0, x0])

def get_sampler(kind: str): return bilinear_sample if 'Bilinear' in kind else nearest_sample

def exy_tensor_from_convention(exy_value: np.ndarray, exy_is_engineering: bool) -> np.ndarray:
    return exy_value * 0.5 if exy_is_engineering else exy_value

def principal_strains_from_components(exx: float, eyy: float, exy: float, *, exy_is_engineering: bool) -> Tuple[float, float]:
    exy_t = exy * (0.5 if exy_is_engineering else 1.0)
    M = np.array([[exx, exy_t], [exy_t, eyy]], dtype=float)
    vals = np.linalg.eigvalsh(M)
    return float(vals[1]), float(vals[0])

def principal_strains_and_directions(exx, eyy, exy, exy_is_engineering: bool):
    exy_t = exy * (0.5 if exy_is_engineering else 1.0)
    M = np.array([[exx, exy_t], [exy_t, eyy]], dtype=float)
    vals, vecs = np.linalg.eigh(M)
    return (vals[1], vals[0]), (vecs[:, 1], vecs[:, 0])

def vm_strain_plane_strain(e1: float, e2: float, e3: float = 0.0) -> float:
    return float(np.sqrt((2.0/9.0) * ((e1 - e2)**2 + (e1 - e3)**2 + (e2 - e3)**2)))

def strain_energy_density_plane_stress(exx: float, eyy: float, exy: float, *, E: float, nu: float, exy_is_engineering: bool) -> float:
    if not exy_is_engineering: gamma = 2.0 * exy
    else: gamma = exy
    fac = E / (1.0 - nu**2)
    sxx = fac * (exx + nu * eyy); syy = fac * (nu * exx + eyy)
    txy = E / (2.0 * (1.0 + nu)) * gamma
    return 0.5 * (exx * sxx + eyy * syy + gamma * txy)

def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    v = np.asarray(values, dtype=float); v = v[np.isfinite(v)]
    if v.size == 0: return float('nan')
    v = v[v >= 0]
    if v.size == 0: return float('nan')
    hist, bin_edges = np.histogram(v, bins=nbins); hist = hist.astype(float)
    prob = hist / hist.sum(); bin_mids = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    w0 = np.cumsum(prob); w1 = 1.0 - w0
    mu = np.cumsum(prob * bin_mids); mu_t = mu[-1]
    sigma_b2 = (mu_t * w0 - mu)**2 / (w0 * w1 + 1e-12)
    idx = int(np.nanargmax(sigma_b2))
    return float(bin_mids[idx])

class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)


class FitGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self._is_panning = False
        self._is_fitted_in_view = True
        self.zoom_factor_base = 1.15
        self.min_zoom = 0.1
        self.max_zoom = 20

    @property
    def current_zoom(self):
        return self.transform().m11()

    def fitInView(self, rect, flags=Qt.KeepAspectRatio):
        super().fitInView(rect, flags)
        self._is_fitted_in_view = True

    def wheelEvent(self, event):
        self._is_fitted_in_view = False
        if event.angleDelta().y() > 0:
            factor = self.zoom_factor_base
            if self.current_zoom >= self.max_zoom:
                return
        else:
            factor = 1 / self.zoom_factor_base
            if self.current_zoom <= self.min_zoom:
                return
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            if self.scene():
                self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        elif event.modifiers() == Qt.ControlModifier and event.button() == Qt.LeftButton:
            self._is_panning = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self._is_panning:
            self._is_panning = False
            self.setDragMode(QGraphicsView.NoDrag)

    def resizeEvent(self, e):
        if self._is_fitted_in_view and self.scene():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(e)

class PointInspectorTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.selA: Optional[Tuple[float, float]] = None
        self.selB: Optional[Tuple[float, float]] = None
        self.image_paths = []
        H = QHBoxLayout(self)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setMinimumWidth(350)
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)

        gb_load = QGroupBox("Load & Settings")
        gl = QGridLayout(gb_load)
        r = 0
        self.btn_load_dic = QPushButton("Load DIC out .bin"); self.btn_load_dic.clicked.connect(self.main_window.load_dic_bin)
        self.btn_load_strain = QPushButton("Load Strain out .bin"); self.btn_load_strain.clicked.connect(self.main_window.load_strain_bin)
        self.btn_load_images = QPushButton("Load Image Sequence"); self.btn_load_images.clicked.connect(self._load_images)
        gl.addWidget(self.btn_load_dic, r, 0, 1, 3); r += 1
        gl.addWidget(self.btn_load_strain, r, 0, 1, 3); r += 1
        gl.addWidget(self.btn_load_images, r, 0, 1, 3); r += 1

        gl.addWidget(QLabel("Units:"), r, 0); self.cb_units = QComboBox(); self.cb_units.addItems(["px","mm","cm","m","in"]) ; gl.addWidget(self.cb_units, r, 1); r += 1
        self.ed_upp = QLineEdit("1.0")
        btn_calibrate = QPushButton("Calibrate...")
        btn_calibrate.clicked.connect(self.open_calibration)
        gl.addWidget(QLabel("Units per pixel:"), r, 0); gl.addWidget(self.ed_upp, r, 1); gl.addWidget(btn_calibrate, r, 2); r += 1

        gl.addWidget(QLabel("Interpolation:"), r, 0); self.cb_interp = QComboBox(); self.cb_interp.addItems(["Bilinear (subpixel)", "Nearest (fast)"]) ; gl.addWidget(self.cb_interp, r, 1, 1, 2); r += 1
        gl.addWidget(QLabel("EXY convention:"), r, 0); self.cb_exyconv = QComboBox(); self.cb_exyconv.addItems(["Tensor εxy", "Engineering γxy"]) ; gl.addWidget(self.cb_exyconv, r, 1, 1, 2); r += 1
        self.cb_compare = QCheckBox("Compare two points in parallel"); self.cb_compare.setChecked(False)
        gl.addWidget(self.cb_compare, r, 0, 1, 3); r += 1

        btn_layout = QHBoxLayout()
        self.btn_save_pts = QPushButton("Save Points"); self.btn_save_pts.clicked.connect(self.save_points)
        self.btn_load_pts = QPushButton("Load Points"); self.btn_load_pts.clicked.connect(self.load_points)
        btn_layout.addWidget(self.btn_save_pts); btn_layout.addWidget(self.btn_load_pts)
        gl.addLayout(btn_layout, r, 0, 1, 3); r += 1

        self.btn_clear = QPushButton("Clear selections"); self.btn_clear.clicked.connect(self.clear_points)
        gl.addWidget(self.btn_clear, r, 0, 1, 3); r += 1
        left.addWidget(gb_load)

        gb_plot_opts = QGroupBox("Plot Content")
        pg = QGridLayout(gb_plot_opts)
        self.cb_disp = QComboBox(); self.cb_disp.addItems(["|U,V|", "U", "V"])
        self.cb_axial = QComboBox(); self.cb_axial.addItems(["Major Principal E1", "Minor Principal E2"])
        self.cb_strain_type = QComboBox(); self.cb_strain_type.addItems(["Engineering ε", "True ln(1+ε)"])
        pg.addWidget(QLabel("Displacement:"), 0, 0); pg.addWidget(self.cb_disp, 0, 1)
        pg.addWidget(QLabel("Axial Strain:"), 1, 0); pg.addWidget(self.cb_axial, 1, 1)
        pg.addWidget(QLabel("Strain Type:"), 2, 0); pg.addWidget(self.cb_strain_type, 2, 1)
        left.addWidget(gb_plot_opts)
        left.addStretch()
        left_scroll.setWidget(left_widget)

        center_widget = QWidget()
        center = QVBoxLayout(center_widget)
        self.scene = QGraphicsScene(self)
        self.view = FitGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.viewport().installEventFilter(self)
        self.frame_slider = QSlider(Qt.Horizontal); self.frame_slider.setEnabled(False)
        self.frame_label = QLabel("Frame: N/A")
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_label)
        center.addLayout(slider_layout)
        center.addWidget(self.view, 1)

        scroll_right = QScrollArea()
        scroll_right.setWidgetResizable(True)
        scroll_right.setFrameShape(QFrame.NoFrame)
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        cols = QHBoxLayout(); right.addLayout(cols, 1)
        self.colA = self._make_plot_column("A"); self.colB = self._make_plot_column("B")
        cols.addLayout(self.colA['layout'], 1); cols.addLayout(self.colB['layout'], 1)
        scroll_right.setWidget(right_widget)

        H.addWidget(left_scroll, 2)
        H.addWidget(center_widget, 3)
        H.addWidget(scroll_right, 5)

        self.cb_disp.currentIndexChanged.connect(self.refresh_plots)
        self.cb_axial.currentIndexChanged.connect(self.refresh_plots)
        self.cb_strain_type.currentIndexChanged.connect(self.refresh_plots)
        self.cb_units.currentIndexChanged.connect(self._apply_units)
        self.ed_upp.editingFinished.connect(self._apply_units)
        self.cb_interp.currentIndexChanged.connect(self._apply_interp)
        self.cb_exyconv.currentIndexChanged.connect(self._apply_exyconv)
        self.frame_slider.valueChanged.connect(self.update_view)
        self._pix_item = None; self._crossA = None; self._crossB = None
        self._labelA = None; self._labelB = None
        self.main_window.ref_img_updated.connect(self.update_view)

    def _load_images(self):
        if iio is None: return QMessageBox.critical(self, "imageio missing", "imageio module not found.")
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Image Sequence (Ref + Deformed)", "", "Images (*.png *.jpg *.tif)")
        if paths:
            self.image_paths = sorted(paths, key=natural_key)
            if self.main_window.ref_img is None and self.image_paths:
                self.main_window.load_ref_img(path=self.image_paths[0])

            self.frame_slider.setRange(0, len(self.image_paths) - 1)
            self.frame_slider.setEnabled(True)
            self.update_view()

    def open_calibration(self):
        img_to_calibrate = None
        if self.image_paths:
            try:
                img_to_calibrate = iio.imread(self.image_paths[0])
                if img_to_calibrate.ndim == 3: 
                    img_to_calibrate = img_to_calibrate[..., 0]  
            except Exception: pass
        elif self.main_window.ref_img is not None:
            img_to_calibrate = self.main_window.ref_img

        if img_to_calibrate is None:
            QMessageBox.warning(self, "Image Missing", "Please load an image sequence or a reference image first.")
            return

        dlg = CalibrateUnitsDialog(img_to_calibrate, self)
        if dlg.exec():
            if dlg.units_per_pixel and dlg.units:
                self.ed_upp.setText(f"{dlg.units_per_pixel:.8f}")
                self.cb_units.setCurrentText(dlg.units)
                self._apply_units()


    def _apply_units(self):
        self.main_window.units = self.cb_units.currentText()
        try: self.main_window.units_per_px = float(self.ed_upp.text())
        except Exception: QMessageBox.warning(self, "Units per pixel", "Enter a valid number.")
        self.refresh_plots()

    def _apply_interp(self):
        self.main_window.interp_kind = self.cb_interp.currentText()
        self.refresh_plots()
    def _apply_exyconv(self):
        self.main_window.exy_is_engineering = (self.cb_exyconv.currentText() == "Engineering γxy")
        self.refresh_plots()

    def _make_plot_column(self, tag: str) -> Dict:
        lay = QVBoxLayout()
        lbl = QLabel(f"Point {tag}: —")
        plot_widget_disp = MplPlotWidget(width=6, height=2.8, dpi=100)
        plot_widget_strn = MplPlotWidget(width=6, height=2.8, dpi=100)
        lay.addWidget(lbl)
        lay.addWidget(plot_widget_disp)
        lay.addWidget(plot_widget_strn)
        return {"layout": lay, "label": lbl, "disp_widget": plot_widget_disp, "strain_widget": plot_widget_strn}

    def _get_displaced_point(self, yx: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float, bool]]:
        if yx is None:
            return None

        ref_y, ref_x = yx
        mw = self.main_window
        frame_idx = self.frame_slider.value()

        if frame_idx == 0 or not mw.U or not self.image_paths:
            return ref_y, ref_x, True

        disp_idx = frame_idx - 1
        if not (0 <= disp_idx < len(mw.U)):
            return ref_y, ref_x, False

        mask = mw.MASK[disp_idx]
        h, w = mask.shape
        is_valid = False
        if 0 <= ref_y < h and 0 <= ref_x < w:
            is_valid = mask[int(round(ref_y)), int(round(ref_x))]

        sampler = get_sampler(mw.interp_kind)
        u = sampler(mw.U[disp_idx], ref_y, ref_x)
        v = sampler(mw.V[disp_idx], ref_y, ref_x)

        return ref_y + v, ref_x + u, is_valid

    def update_view(self):
        image_to_show = None
        frame_idx = self.frame_slider.value()

        if self.image_paths and 0 <= frame_idx < len(self.image_paths):
            try:
                im = iio.imread(self.image_paths[frame_idx])
                if im.ndim == 3: im = im[..., 0]
                image_to_show = np.clip(im, 0, 255).astype(np.uint8)
                if frame_idx == 0:
                    self.frame_label.setText("Ref")
                else:
                    self.frame_label.setText(f"{frame_idx}/{len(self.image_paths) - 1}")
            except Exception: pass
        else:
            image_to_show = self.main_window.ref_img
            self.frame_label.setText("Ref")

        if image_to_show is None: return

        pix = QPixmap.fromImage(np_to_qimage(image_to_show))
        if self._pix_item is None: self._pix_item = self.scene.addPixmap(pix)
        else: self._pix_item.setPixmap(pix)

        self.scene.setSceneRect(QRectF(pix.rect()))
        if self.view._is_fitted_in_view:
             self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        pt_A_info = self._get_displaced_point(self.selA)
        pt_B_info = self._get_displaced_point(self.selB)

        self._draw_cross(pt_A_info, "A")
        self._draw_cross(pt_B_info, "B")

    def eventFilter(self, obj, ev):
        if obj is self.view.viewport() and ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.LeftButton:
            pt = self.view.mapToScene(ev.pos()); y, x = float(pt.y()), float(pt.x())
            self.pick_point((y, x)); return True
        return super().eventFilter(obj, ev)

    def clear_points(self):
        self.selA = None; self.selB = None
        self._draw_cross(None, "A"); self._draw_cross(None, "B")
        self.refresh_plots()

    def pick_point(self, yx: Tuple[float,float]):
        if not self.cb_compare.isChecked(): self.selA = yx; self.selB = None
        else:
            if self.selA is None or (self.selA and self.selB): self.selA = yx; self.selB = None
            elif self.selB is None: self.selB = yx

        self.update_view()
        self.refresh_plots()

    def _draw_cross(self, point_info: Optional[Tuple[float,float,bool]], tag: str):
        target_cross_attr = "_crossA" if tag == "A" else "_crossB"
        target_label_attr = "_labelA" if tag == "A" else "_labelB"

        target_cross = getattr(self, target_cross_attr, None)
        target_label = getattr(self, target_label_attr, None)

        if target_cross:
            for item in target_cross:
                if item and item.scene() == self.scene: self.scene.removeItem(item)
        if target_label and target_label.scene() == self.scene:
            self.scene.removeItem(target_label)

        setattr(self, target_cross_attr, [])
        setattr(self, target_label_attr, None)

        if point_info is not None:
            y, x, is_valid = point_info

            color = QColor(128, 128, 128)
            if is_valid:
                color = QColor(255,0,0) if tag == "A" else QColor(0,120,255)

            pen = QPen(color, 1.6); pen.setCosmetic(True)

            new_cross = [
                self.scene.addLine(x-8, y, x+8, y, pen),
                self.scene.addLine(x, y-8, x, y+8, pen)
            ]
            setattr(self, target_cross_attr, new_cross)

            if not is_valid:
                font = QFont("Arial", 8)
                label = self.scene.addText("Outside ROI", font)
                label.setDefaultTextColor(QColor("yellow"))
                label.setPos(x + 5, y - 15)
                setattr(self, target_label_attr, label)

    def refresh_plots(self):
        self._refresh_one(self.colA, self.selA, "A")
        self._refresh_one(self.colB, self.selB, "B")

    def _refresh_one(self, col, yx, tag: str):
        mw = self.main_window
        disp_widget = col['disp_widget']
        strain_widget = col['strain_widget']

        col['label'].setText(f"Point {tag}: —" if yx is None else f"Point {tag}: y={yx[0]:.1f}, x={yx[1]:.1f} (ref)")

        if mw.n_frames == 0 or yx is None:
            disp_widget.set_data({}, meta={'title': "Displacement", 'xlabel': "Frame"})
            strain_widget.set_data({}, meta={'title': "Strain", 'xlabel': "Frame"})
            return

        sampler = get_sampler(mw.interp_kind)
        t = np.arange(1, mw.n_frames + 1, dtype=float)
        U_series, V_series, A_series, T_series = [], [], [], []
        for i in range(mw.n_frames):
            u = sampler(mw.U[i], yx[0], yx[1]) if i < len(mw.U) else np.nan
            v = sampler(mw.V[i], yx[0], yx[1]) if i < len(mw.V) else np.nan
            U_series.append(u * mw.units_per_px); V_series.append(v * mw.units_per_px)
            exx = sampler(mw.EXX[i], yx[0], yx[1]) if i < len(mw.EXX) else np.nan
            eyy = sampler(mw.EYY[i], yx[0], yx[1]) if i < len(mw.EYY) else np.nan
            exy = sampler(mw.EXY[i], yx[0], yx[1]) if i < len(mw.EXY) else 0.0
            e1, e2 = principal_strains_from_components(exx, eyy, exy, exy_is_engineering=mw.exy_is_engineering)
            if self.cb_axial.currentIndex() == 0: A_series.append(e1); T_series.append(e2)
            else: A_series.append(e2); T_series.append(e1)

        U_series = np.asarray(U_series); V_series = np.asarray(V_series)
        disp_mag = np.sqrt(U_series**2 + V_series**2)

        disp_data = {'x': t}; disp_meta = {'xlabel': "Frame", 'ylabel': f"Disp [{mw.units}]", 'title': f"Displacement at Point {tag}"}
        choice = self.cb_disp.currentText()
        if choice == "|U,V|": disp_data['|U,V|'] = disp_mag
        elif choice == "U": disp_data['U'] = U_series
        else: disp_data['V'] = V_series
        disp_widget.set_data(disp_data, disp_meta)

        A_series = np.asarray(A_series); T_series = np.asarray(T_series)
        if self.cb_strain_type.currentIndex() == 1:
            A_plot = np.log1p(A_series); T_plot = np.log1p(T_series)
            strain_ylabel = "True Strain"
        else:
            A_plot = A_series; T_plot = T_series
            strain_ylabel = "Eng. Strain"

        strain_data = {'x': t, 'Axial': A_plot, 'Transverse': T_plot}
        strain_meta = {'xlabel': "Frame", 'ylabel': strain_ylabel, 'title': f"Strain at Point {tag}"}
        strain_widget.set_data(strain_data, strain_meta)

    def save_points(self):
        if self.selA is None and self.selB is None:
            QMessageBox.information(self, "No Points", "No points have been selected to save.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Selected Points", "", "Text Files (*.txt)")
        if not path: return

        try:
            with open(path, 'w') as f:
                if self.selA: f.write(f"A: {self.selA[0]},{self.selA[1]}\n")
                if self.selB: f.write(f"B: {self.selB[0]},{self.selB[1]}\n")
            QMessageBox.information(self, "Success", f"Points saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save points file:\n{e}")

    def load_points(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Selected Points", "", "Text Files (*.txt)")
        if not path: return
        try:
            with open(path, 'r') as f:
                lines = f.readlines(); loaded_A, loaded_B = None, None
                for line in lines:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        tag, coords_str = parts
                        coords = [float(c.strip()) for c in coords_str.split(',')]
                        if len(coords) == 2:
                            if tag.strip().upper() == 'A': loaded_A = (coords[0], coords[1])
                            elif tag.strip().upper() == 'B': loaded_B = (coords[0], coords[1])
                self.selA = loaded_A; self.selB = loaded_B
                self.update_view(); self.refresh_plots()
                QMessageBox.information(self, "Success", f"Points loaded from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load or parse points file:\n{e}")


class RelativeDisplacementTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.selA: Optional[Tuple[float,float]] = None
        self.selB: Optional[Tuple[float,float]] = None
        self.image_paths = []
        H = QHBoxLayout(self)

        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True); left_scroll.setFrameShape(QFrame.NoFrame)
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)

        gb_load = QGroupBox("1. Load Data")
        gl = QGridLayout(gb_load)
        self.btn_load_dic = QPushButton("Load DIC out .bin"); self.btn_load_dic.clicked.connect(self.main_window.load_dic_bin)
        self.btn_load_images = QPushButton("Load Image Sequence"); self.btn_load_images.clicked.connect(self._load_images)
        self.image_list_label = QLabel("0 images loaded.")
        gl.addWidget(self.btn_load_dic, 0, 0, 1, 2)
        gl.addWidget(self.btn_load_images, 1, 0, 1, 2)
        gl.addWidget(self.image_list_label, 2, 0, 1, 2)
        left.addWidget(gb_load)

        gb_select_info = QGroupBox("2. Selections")
        sl = QVBoxLayout(gb_select_info)
        self.info_label = QLabel("Click two points on the image preview.")
        self.btn_clear = QPushButton("Clear Selections"); self.btn_clear.clicked.connect(self.clear_points)
        sl.addWidget(self.info_label); sl.addWidget(self.btn_clear)
        left.addWidget(gb_select_info)

        gb_opts = QGroupBox("3. Method & Options")
        ol = QGridLayout(gb_opts); r = 0
        self.cb_method = QComboBox(); self.cb_method.addItems(["Distance |xB-xA|", "Extension ΔL = |xB-xA|-|X_B-X_A|", "Tangential δ_t (along AB)", "Normal δ_n (⊥ AB)"])
        ol.addWidget(QLabel("Method:"), r, 0); ol.addWidget(self.cb_method, r, 1); r += 1
        self.cb_interp = QComboBox(); self.cb_interp.addItems(["Bilinear (subpixel)", "Nearest (fast)"])
        ol.addWidget(QLabel("Interpolation:"), r, 0); ol.addWidget(self.cb_interp, r, 1); r += 1
        left.addWidget(gb_opts)

        gb_calib = QGroupBox("4. Calibration")
        calib_layout = QGridLayout(gb_calib)
        self.ed_upp = QLineEdit("1.0")
        self.cb_units = QComboBox(); self.cb_units.addItems(["px","mm","cm","m","in"])
        btn_calibrate = QPushButton("Calibrate...")
        btn_calibrate.clicked.connect(self.open_calibration)
        calib_layout.addWidget(QLabel("Units per pixel:"), 0, 0); calib_layout.addWidget(self.ed_upp, 0, 1)
        calib_layout.addWidget(QLabel("Units:"), 1, 0); calib_layout.addWidget(self.cb_units, 1, 1)
        calib_layout.addWidget(btn_calibrate, 0, 2, 2, 1)
        left.addWidget(gb_calib)

        gb_plot = QGroupBox("Relative Metric vs. Time")
        pl = QVBoxLayout(gb_plot)
        self.plot_widget = MplPlotWidget()
        pl.addWidget(self.plot_widget)
        left.addWidget(gb_plot)
        left.addStretch()
        left_scroll.setWidget(left_widget)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.scene = QGraphicsScene(self); self.view = FitGraphicsView(self)
        self.view.setScene(self.scene); self.view.viewport().installEventFilter(self)

        self.frame_slider = QSlider(Qt.Horizontal); self.frame_slider.setEnabled(False)
        self.frame_label = QLabel("Frame: N/A")
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_label)

        right_layout.addLayout(slider_layout)
        right_layout.addWidget(self.view, 1)

        H.addWidget(left_scroll, 1)
        H.addWidget(right_panel, 2)

        self._pix_item = None; self._crossA = None; self._crossB = None
        self.ed_upp.editingFinished.connect(self._apply_units)
        self.cb_units.currentIndexChanged.connect(self._apply_units)
        self.cb_method.currentIndexChanged.connect(self.plot_relative_distance)
        self.frame_slider.valueChanged.connect(self.update_view)
        self._apply_units()

    def _load_images(self):
        if iio is None: return QMessageBox.critical(self, "imageio missing", "imageio module not found.")
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Image Sequence (Ref + Deformed)", "", "Images (*.png *.jpg *.tif)")
        if paths:
            self.image_paths = sorted(paths, key=natural_key)
            self.image_list_label.setText(f"{len(self.image_paths)} images loaded.")

            if self.main_window.ref_img is None and self.image_paths:
                self.main_window.load_ref_img(path=self.image_paths[0])

            num_images = len(self.image_paths)
            if num_images > 0:
                self.frame_slider.setRange(0, num_images - 1)
                self.frame_slider.setValue(0)
                self.frame_slider.setEnabled(True)
            else:
                self.frame_slider.setEnabled(False)
            self.update_view()

    def open_calibration(self):
        img_to_calibrate = None
        if self.image_paths:
            try:
                img_to_calibrate = iio.imread(self.image_paths[0])
                if img_to_calibrate.ndim == 3: 
                    img_to_calibrate = img_to_calibrate[..., 0]  
            except Exception: pass
        elif self.main_window.ref_img is not None:
            img_to_calibrate = self.main_window.ref_img

        if img_to_calibrate is None:
            QMessageBox.warning(self, "Image Missing", "Please load an image sequence or a reference image first.")
            return

        dlg = CalibrateUnitsDialog(img_to_calibrate, self)
        if dlg.exec():
            if dlg.units_per_pixel and dlg.units:
                self.ed_upp.setText(f"{dlg.units_per_pixel:.8f}")
                self.cb_units.setCurrentText(dlg.units)
                self._apply_units()


    def _apply_units(self):
        mw = self.main_window
        try:
            mw.units_per_px = float(self.ed_upp.text())
            mw.units = self.cb_units.currentText()
            if hasattr(mw, 'tab1'):
                mw.tab1.ed_upp.setText(f"{mw.units_per_px}")
                mw.tab1.cb_units.setCurrentText(mw.units)
            if self.selA and self.selB: self.plot_relative_distance()
        except (ValueError, TypeError): QMessageBox.warning(self, "Invalid Input", "Units per pixel must be a valid number.")

    def eventFilter(self, obj, ev):
        if obj is self.view.viewport() and ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.LeftButton:
            pt = self.view.mapToScene(ev.pos()); self.pick_point((float(pt.y()), float(pt.x()))); return True
        return super().eventFilter(obj, ev)

    def pick_point(self, yx):
        if self.selA is None: self.selA = yx; self.info_label.setText(f"Point A: ({yx[1]:.1f}, {yx[0]:.1f}). Select Point B.")
        elif self.selB is None: self.selB = yx; self.info_label.setText(f"A:({self.selA[1]:.1f}, {self.selA[0]:.1f}), B:({yx[1]:.1f}, {yx[0]:.1f})"); self.plot_relative_distance()
        else: self.clear_points(); self.selA = yx; self.info_label.setText(f"Point A: ({yx[1]:.1f}, {yx[0]:.1f}). Select Point B.")
        self.update_view()

    def _get_displaced_point(self, yx: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float, bool]]:
        if yx is None:
            return None

        ref_y, ref_x = yx
        mw = self.main_window
        frame_idx = self.frame_slider.value()

        if frame_idx == 0 or not mw.U or not self.image_paths:
            return ref_y, ref_x, True

        disp_idx = frame_idx - 1
        if not (0 <= disp_idx < len(mw.U)):
            return ref_y, ref_x, False

        mask = mw.MASK[disp_idx]
        h, w = mask.shape
        is_valid = False
        if 0 <= ref_y < h and 0 <= ref_x < w:
            is_valid = mask[int(round(ref_y)), int(round(ref_x))]

        sampler = get_sampler(self.cb_interp.currentText())
        u = sampler(mw.U[disp_idx], ref_y, ref_x)
        v = sampler(mw.V[disp_idx], ref_y, ref_x)

        return ref_y + v, ref_x + u, is_valid

    def update_view(self):
        frame_idx = self.frame_slider.value()
        image_to_show = None

        if self.image_paths and 0 <= frame_idx < len(self.image_paths):
            try:
                im = iio.imread(self.image_paths[frame_idx])
                if im.ndim == 3: im = im[..., 0]
                image_to_show = np.clip(im, 0, 255).astype(np.uint8)
                if frame_idx == 0: self.frame_label.setText("Ref")
                else: self.frame_label.setText(f"{frame_idx}/{len(self.image_paths) - 1}")
            except Exception:
                self.frame_label.setText("Error")
        elif self.main_window.ref_img is not None:
             image_to_show = self.main_window.ref_img
             self.frame_label.setText("Ref")

        if image_to_show is None: return

        pix = QPixmap.fromImage(np_to_qimage(image_to_show))
        if self._pix_item is None: self._pix_item = self.scene.addPixmap(pix)
        else: self._pix_item.setPixmap(pix)

        self.scene.setSceneRect(QRectF(pix.rect()))
        if self.view._is_fitted_in_view:
             self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        pt_A_info = self._get_displaced_point(self.selA)
        pt_B_info = self._get_displaced_point(self.selB)

        if self._crossA and self._crossA.scene() == self.scene: self.scene.removeItem(self._crossA)
        self._crossA = None
        if self._crossB and self._crossB.scene() == self.scene: self.scene.removeItem(self._crossB)
        self._crossB = None

        if pt_A_info:
            y, x, is_valid = pt_A_info
            color = QColor("red") if is_valid else QColor(128, 128, 128)
            pen = QPen(color, 1.8); pen.setCosmetic(True)
            self._crossA = self.scene.addEllipse(x-5, y-5, 10, 10, pen)

        if pt_B_info:
            y, x, is_valid = pt_B_info
            color = QColor("blue") if is_valid else QColor(128, 128, 128)
            pen = QPen(color, 1.8); pen.setCosmetic(True)
            self._crossB = self.scene.addEllipse(x-5, y-5, 10, 10, pen)

    def clear_points(self):
        self.selA, self.selB = None, None; self.info_label.setText("Click two points on the image.")
        self.plot_widget.set_data({}, meta={'title': "Relative Displacement"})
        self.update_view()

    def plot_relative_distance(self):
        mw = self.main_window
        if not all([self.selA, self.selB, mw.dic_out, len(mw.U) > 0]):
            self.plot_widget.set_data({}, meta={'title': "Relative Displacement"})
            return

        sampler = get_sampler(self.cb_interp.currentText())
        def get_series(yx):
            U_series, V_series = [], []
            for i in range(mw.n_frames):
                u = sampler(mw.U[i], yx[0], yx[1]); v = sampler(mw.V[i], yx[0], yx[1])
                U_series.append(u * mw.units_per_px); V_series.append(v * mw.units_per_px)
            return np.array(U_series), np.array(V_series)
        uA, vA = get_series(self.selA); uB, vB = get_series(self.selB)
        XA = np.array([self.selA[1], self.selA[0]]) * mw.units_per_px
        XB = np.array([self.selB[1], self.selB[0]]) * mw.units_per_px
        AB0 = XB - XA; L0 = float(np.linalg.norm(AB0))
        if L0 == 0: QMessageBox.warning(self, "Points coincide", "Pick two distinct points."); return
        t_hat = AB0 / L0; n_hat = np.array([-t_hat[1], t_hat[0]])
        values = []; labels = {0: "Distance |xB-xA|", 1: "Extension ΔL", 2: "Tangential δ_t", 3: "Normal δ_n"}
        method_idx = self.cb_method.currentIndex()
        for i in range(mw.n_frames):
            xA = XA + np.array([uA[i], vA[i]]); xB = XB + np.array([uB[i], vB[i]]); d = xB - xA
            if method_idx == 0: values.append(np.linalg.norm(d))
            elif method_idx == 1: values.append(np.linalg.norm(d) - L0)
            elif method_idx == 2: values.append(float(np.dot(d, t_hat)))
            else: values.append(float(np.dot(d, n_hat)))

        data = {'x': np.arange(1, mw.n_frames + 1), 'Metric': np.array(values)}
        meta = {'xlabel': "Frame Number", 'ylabel': f"[{mw.units}]", 'title': labels[method_idx]}
        self.plot_widget.set_data(data, meta)

class SDICalculatorTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.sdi_values = []; self.sdi_fields = []; self.sdi_masks = []
        self.last_metric_name = "Field"
        self.global_vmin = None
        self.global_vmax = None

        H = QHBoxLayout(self)

        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True); left_scroll.setFrameShape(QFrame.NoFrame)
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)

        gb_load = QGroupBox("1. Load Strain Data"); gl = QGridLayout(gb_load)
        self.btn_load_strain = QPushButton("Load Strain out .bin"); self.btn_load_strain.clicked.connect(self.main_window.load_strain_bin)
        gl.addWidget(self.btn_load_strain, 0, 0, 1, 2); left.addWidget(gb_load)
        gb_method = QGroupBox("2. SDI Metric & Parameters"); ml = QGridLayout(gb_method)
        self.cb_metric = QComboBox()
        self.cb_metric.addItems(["mean |E1| (principal)", "p‑percentile |E1|", "area fraction |E1| > T (Otsu if T=auto)", "mean von Mises eq. strain (plane‑strain)", "mean strain energy density (plane‑stress)"])
        self.cb_metric.setItemData(0, "Calculates the mean of the absolute value of the first principal strain.", Qt.ToolTipRole)
        self.cb_metric.setItemData(1, "Calculates the p-th percentile of the absolute value of the first principal strain.", Qt.ToolTipRole)
        self.cb_metric.setItemData(2, "Calculates the fraction of the area where |E1| exceeds a threshold T.", Qt.ToolTipRole)
        self.cb_metric.setItemData(3, "Calculates the mean von Mises equivalent strain, assuming plane-strain conditions.", Qt.ToolTipRole)
        self.cb_metric.setItemData(4, "Calculates the mean strain energy density, assuming plane-stress conditions.", Qt.ToolTipRole)

        self.sp_p = QDoubleSpinBox(); self.sp_p.setRange(0.0, 100.0); self.sp_p.setDecimals(1); self.sp_p.setValue(95.0)
        self.ed_T = QLineEdit("auto")
        self.sp_E = QDoubleSpinBox(); self.sp_E.setRange(1e-3, 1e6); self.sp_E.setDecimals(3); self.sp_E.setValue(210000.0)
        self.sp_nu = QDoubleSpinBox(); self.sp_nu.setRange(0.0, 0.499); self.sp_nu.setDecimals(3); self.sp_nu.setValue(0.30)
        self.cb_exyconv = QComboBox(); self.cb_exyconv.addItems(["Tensor εxy", "Engineering γxy"])
        self.chk_outliers = QCheckBox("Remove Outliers (±3σ)"); self.chk_outliers.setChecked(False)

        r=0; ml.addWidget(QLabel("Metric:"), r, 0); ml.addWidget(self.cb_metric, r, 1); r += 1
        ml.addWidget(QLabel("Percentile p [%]:"), r, 0); ml.addWidget(self.sp_p, r, 1); r += 1
        ml.addWidget(QLabel("Threshold T (strain):"), r, 0); ml.addWidget(self.ed_T, r, 1); r += 1
        ml.addWidget(QLabel("Young's E:"), r, 0); ml.addWidget(self.sp_E, r, 1); r += 1
        ml.addWidget(QLabel("Poisson ν:"), r, 0); ml.addWidget(self.sp_nu, r, 1); r += 1
        ml.addWidget(QLabel("EXY convention:"), r, 0); ml.addWidget(self.cb_exyconv, r, 1); r += 1
        ml.addWidget(self.chk_outliers, r, 0, 1, 2); r+=1
        left.addWidget(gb_method)

        gb_actions = QGroupBox("3. Actions")
        action_layout = QHBoxLayout(gb_actions)
        self.btn_calculate = QPushButton("Calculate SDI"); self.btn_calculate.clicked.connect(self.calculate_sdi)
        self.btn_export = QPushButton("Export SDI to .csv"); self.btn_export.clicked.connect(self.export_csv)
        action_layout.addWidget(self.btn_calculate); action_layout.addWidget(self.btn_export)
        left.addWidget(gb_actions)

        gb_plot = QGroupBox("SDI vs. Time"); pl = QVBoxLayout(gb_plot)
        self.sdi_plot_widget = MplPlotWidget(height=2.5)
        pl.addWidget(self.sdi_plot_widget)
        left.addWidget(gb_plot)
        left.addStretch()
        left_scroll.setWidget(left_widget)

        right = QVBoxLayout()
        gb_preview = QGroupBox("Data Field Preview"); prev_layout = QGridLayout(gb_preview)

        self.preview_plot_widget = MplPlotWidget(height=3)
        self.preview_ax = self.preview_plot_widget.ax
        self.preview_fig = self.preview_plot_widget.fig

        self.frame_slider = QSlider(Qt.Horizontal); self.frame_slider.setEnabled(False)
        self.frame_label = QLabel("Frame: N/A")
        self.cb_cmap = QComboBox(); self.cb_cmap.addItems(["jet", "viridis", "turbo", "plasma", "inferno"])
        prev_layout.addWidget(self.preview_plot_widget, 0, 0, 1, 3)
        prev_layout.addWidget(QLabel("Frame:"), 1, 0); prev_layout.addWidget(self.frame_slider, 1, 1); prev_layout.addWidget(self.frame_label, 1, 2)
        prev_layout.addWidget(QLabel("Colormap:"), 2, 0); prev_layout.addWidget(self.cb_cmap, 2, 1, 1, 2)
        right.addWidget(gb_preview)

        H.addWidget(left_scroll, 1)
        H.addLayout(right, 2)

        self.frame_slider.valueChanged.connect(self.update_plots)
        self.cb_cmap.currentIndexChanged.connect(self.update_plots)

    def calculate_sdi(self):
        mw = self.main_window
        if not all([mw.strain_out, len(mw.EXX) > 0]): QMessageBox.warning(self, "Incomplete Data", "Load Strain data first."); return
        metric = self.cb_metric.currentText(); exy_is_engineering = (self.cb_exyconv.currentText() == "Engineering γxy")
        E = float(self.sp_E.value()); nu = float(self.sp_nu.value())
        self.sdi_values = []; self.sdi_fields = []; self.sdi_masks = []
        all_field_vals = []

        prog = QProgressDialog("Calculating SDI...", "Cancel", 0, mw.n_frames, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)

        for i in range(mw.n_frames):
            prog.setValue(i)
            if prog.wasCanceled(): break

            exx, eyy, exy = mw.EXX[i], mw.EYY[i], mw.EXY[i]
            mask = mw.MASK[i] if i < len(mw.MASK) else np.ones_like(exx, dtype=bool)
            self.sdi_masks.append(mask)

            exy_t = exy_tensor_from_convention(exy, exy_is_engineering)
            tr = exx + eyy; rad = np.sqrt(((exx - eyy) * 0.5)**2 + exy_t**2)
            e1, e2 = (tr * 0.5) + rad, (tr * 0.5) - rad

            current_field, sdi = None, np.nan

            def get_valid_data(field_data):
                data = field_data[mask]
                if self.chk_outliers.isChecked() and data.size > 10:
                    mean = np.nanmean(data); std = np.nanstd(data)
                    if std > 1e-9:
                        return data[np.abs(data - mean) < 3 * std]
                return data

            if "E1" in metric:
                field = np.abs(e1); current_field, self.last_metric_name = field, "|E1|"
                vals = get_valid_data(field)
                if vals.size == 0: sdi = np.nan
                elif metric.startswith("mean"): sdi = float(np.nanmean(vals))
                elif metric.startswith("p‑percentile"): sdi = float(np.nanpercentile(vals, float(self.sp_p.value())))
                else:
                    try: T = otsu_threshold(vals) if self.ed_T.text().strip().lower() == "auto" else float(self.ed_T.text())
                    except (ValueError, TypeError): QMessageBox.warning(self, "Threshold Error", "Invalid threshold value."); prog.cancel(); return
                    sdi = float(np.count_nonzero(vals > T) / max(1, vals.size))
            elif "von Mises" in metric:
                field = np.sqrt((2.0/9.0) * ((e1 - e2)**2 + e1**2 + e2**2)); current_field, self.last_metric_name = field, "Von Mises Strain"
                vals = get_valid_data(field)
                if vals.size > 0: sdi = float(np.nanmean(vals))
            else:
                gamma = 2.0 * exy_t; fac = E / (1.0 - nu**2); sxx = fac * (exx + nu * eyy); syy = fac * (nu * exx + eyy)
                txy = E / (2.0 * (1.0 + nu)) * gamma; W = 0.5 * (exx * sxx + eyy * syy + gamma * txy)
                current_field, self.last_metric_name = W, "Strain Energy Density"
                vals = get_valid_data(W)
                if vals.size > 0: sdi = float(np.nanmean(vals))

            self.sdi_values.append(sdi); self.sdi_fields.append(current_field)
            if current_field is not None: all_field_vals.extend(current_field[mask & np.isfinite(current_field)])

        prog.setValue(mw.n_frames)

        if all_field_vals:
            self.global_vmin = float(np.percentile(all_field_vals, 2))
            self.global_vmax = float(np.percentile(all_field_vals, 98))
        else: self.global_vmin, self.global_vmax = None, None

        self.frame_slider.setRange(0, mw.n_frames - 1); self.frame_slider.setValue(mw.n_frames - 1)
        self.frame_slider.setEnabled(True); self.update_plots()

    def update_plots(self):
        if not self.sdi_values: return

        data = {'x': np.arange(1, len(self.sdi_values) + 1), 'SDI': np.array(self.sdi_values)}
        meta = {'xlabel': "Frame Number", 'ylabel': "SDI", 'title': "SDI Over Time"}
        self.sdi_plot_widget.set_data(data, meta)

        frame_idx = self.frame_slider.value()
        if frame_idx >= len(self.sdi_fields): return

        self.frame_label.setText(f"{frame_idx + 1} / {len(self.sdi_fields)}")
        field_to_plot = self.sdi_fields[frame_idx].copy(); mask = self.sdi_masks[frame_idx]
        field_to_plot[~mask] = np.nan

        self.preview_fig.clear()
        self.preview_ax = self.preview_fig.add_subplot(111)

        if np.all(np.isnan(field_to_plot)):
             self.preview_ax.text(0.5, 0.5, 'No valid data to display', ha='center', va='center')
        else:
            im = self.preview_ax.imshow(field_to_plot, cmap=self.cb_cmap.currentText(), origin='upper', vmin=self.global_vmin, vmax=self.global_vmax)
            divider = make_axes_locatable(self.preview_ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            self.preview_fig.colorbar(im, cax=cax)
            self.preview_ax.set_title(f"{self.last_metric_name} (Frame {frame_idx + 1})")
        self.preview_ax.axis('off')
        try:
            self.preview_fig.tight_layout()
        except RuntimeError:
            pass 
        self.preview_plot_widget.canvas.draw()

    def export_csv(self):
        if not self.sdi_values:
            QMessageBox.warning(self, "No Data", "Please calculate SDI first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save SDI Data", "sdi_timeseries.csv", "CSV Files (*.csv)")
        if not path: return

        try:
            frames = np.arange(1, len(self.sdi_values) + 1)
            data_to_save = np.vstack((frames, np.array(self.sdi_values))).T
            np.savetxt(path, data_to_save, delimiter=',', header='Frame,SDI', comments='')
            QMessageBox.information(self, "Success", f"SDI data exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save file: {e}")


class PoissonRatioTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.selected_point: Optional[Tuple[float, float]] = None
        self.selected_roi: Optional[QRectF] = None
        self.image_paths = [] 

        H = QHBoxLayout(self)

        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True); left_scroll.setFrameShape(QFrame.NoFrame)
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)

        gb_load = QGroupBox("1. Load Data"); gl = QGridLayout(gb_load)
        self.btn_load_strain = QPushButton("Load Strain out .bin"); self.btn_load_strain.clicked.connect(self.main_window.load_strain_bin)
        self.btn_load_ref = QPushButton("Load Reference Image"); self.btn_load_ref.clicked.connect(lambda: self.main_window.load_ref_img())
        self.btn_load_sequence = QPushButton("Load Image Sequence")
        self.btn_load_sequence.clicked.connect(self._load_images)
        gl.addWidget(self.btn_load_strain, 0, 0, 1, 2)
        gl.addWidget(self.btn_load_ref, 1, 0)
        gl.addWidget(self.btn_load_sequence, 1, 1)
        left.addWidget(gb_load)

        gb_select_info = QGroupBox("2. Selection"); sl = QFormLayout(gb_select_info)
        self.cb_selection_mode = QComboBox(); self.cb_selection_mode.addItems(["Single Point", "Rectangular Region (ROI)"])
        self.info_label = QLabel("Select a point or region on the image.")
        sl.addRow("Selection Method:", self.cb_selection_mode)
        sl.addRow(self.info_label)
        left.addWidget(gb_select_info)

        gb_opts = QGroupBox("3. Options"); ol = QFormLayout(gb_opts)
        self.cb_axial = QComboBox(); self.cb_axial.addItems(["Major Principal (E1)", "Minor Principal (E2)"])
        self.cb_interp = QComboBox(); self.cb_interp.addItems(["Bilinear (subpixel)", "Nearest (fast)"])
        frame_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal); self.frame_slider.setEnabled(False)
        self.frame_label = QLabel("N/A")
        frame_layout.addWidget(self.frame_slider); frame_layout.addWidget(self.frame_label)
        ol.addRow("Axial Strain:", self.cb_axial)
        ol.addRow("Interpolation:", self.cb_interp)
        ol.addRow("Preview Frame:", frame_layout)
        left.addWidget(gb_opts)

        gb_plot = QGroupBox("Poisson's Ratio (ν = -E_trans / E_axial) vs. Time"); pl = QVBoxLayout(gb_plot)
        self.plot_widget = MplPlotWidget()
        pl.addWidget(self.plot_widget)
        left.addWidget(gb_plot)
        left.addStretch()
        left_scroll.setWidget(left_widget)

        self.scene = QGraphicsScene(self)
        self.view = FitGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.viewport().installEventFilter(self)
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.view)

        H.addWidget(left_scroll, 1)
        H.addWidget(self.view, 2)

        self._pix_item = None; self._selection_graphic = None
        self.cb_selection_mode.currentIndexChanged.connect(self.clear_selection)
        self.cb_interp.currentIndexChanged.connect(self.calculate_and_plot)
        self.cb_axial.currentIndexChanged.connect(self.calculate_and_plot)
        self.frame_slider.valueChanged.connect(self.load_base_image)
        self.main_window.ref_img_updated.connect(self.load_base_image)


    def _load_images(self):
        if iio is None: return QMessageBox.critical(self, "imageio missing", "imageio module not found.")
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Image Sequence (Ref + Deformed)", "", "Images (*.png *.jpg *.tif)")
        if paths:
            self.image_paths = sorted(paths, key=natural_key)

            if self.main_window.ref_img is None:
                self.main_window.load_ref_img(path=self.image_paths[0])

            num_images = len(self.image_paths)
            if num_images > 0:
                self.frame_slider.setRange(0, num_images - 1)
                self.frame_slider.setValue(0)
                self.frame_slider.setEnabled(True)
            else:
                self.frame_slider.setEnabled(False)
            
            self.load_base_image() 

    def eventFilter(self, obj, ev):
        if obj is self.view.viewport():
            if self.cb_selection_mode.currentText() == "Single Point":
                if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.LeftButton:
                    pt = self.view.mapToScene(ev.pos()); self.pick_point(pt); return True
            else:
                if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.LeftButton:
                    self.origin = ev.pos(); self.rubber_band.setGeometry(QRectF(self.origin, QPointF()).toRect()); self.rubber_band.show(); return True
                elif ev.type() == QEvent.Type.MouseMove and ev.buttons() & Qt.LeftButton:
                    self.rubber_band.setGeometry(QRectF(self.origin, ev.pos()).toRect()); return True
                elif ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.LeftButton:
                    self.rubber_band.hide(); self.pick_roi(self.view.mapToScene(self.origin), self.view.mapToScene(ev.pos())); return True
        return super().eventFilter(obj, ev)

    def clear_selection(self):
        self.selected_point = None; self.selected_roi = None
        if self._selection_graphic and self._selection_graphic.scene() == self.scene:
            self.scene.removeItem(self._selection_graphic)
        self._selection_graphic = None
        self.info_label.setText("Select a point or region on the image.")
        self.calculate_and_plot()

    def pick_point(self, pt: QPointF):
        self.clear_selection()
        self.selected_point = (pt.y(), pt.x())
        self.info_label.setText(f"Point: y={pt.y():.1f}, x={pt.x():.1f}")
        self.update_view_overlays(); self.calculate_and_plot()

    def pick_roi(self, p1: QPointF, p2: QPointF):
        self.clear_selection()
        self.selected_roi = QRectF(p1, p2).normalized()
        self.info_label.setText(f"ROI: ({self.selected_roi.left():.1f}, {self.selected_roi.top():.1f}) to ({self.selected_roi.right():.1f}, {self.selected_roi.bottom():.1f})")
        self.update_view_overlays(); self.calculate_and_plot()


    def load_base_image(self):
        mw = self.main_window
        image_to_show = None
        frame_idx = self.frame_slider.value()
    
        if self.image_paths and 0 <= frame_idx < len(self.image_paths):
            try:
                im = iio.imread(self.image_paths[frame_idx])
                if im.ndim == 3: im = im[..., 0]
                image_to_show = np.clip(im, 0, 255).astype(np.uint8)
                if frame_idx == 0:
                    self.frame_label.setText("Ref")
                else:
                    self.frame_label.setText(f"{frame_idx}/{len(self.image_paths) - 1}")
            except Exception:
                self.frame_label.setText("Error")
        elif mw.ref_img is not None:
            image_to_show = mw.ref_img
            self.frame_label.setText("Ref")
        
        if image_to_show is None:
            return
    
        self.scene.clear()
        self._selection_graphic = None
        
        pix = QPixmap.fromImage(np_to_qimage(image_to_show))
        self._pix_item = self.scene.addPixmap(pix)
        self.scene.setSceneRect(QRectF(pix.rect()))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
        if mw.n_frames > 0 and not self.image_paths:
            self.frame_slider.setRange(0, mw.n_frames - 1)
            self.frame_slider.setEnabled(True)
        
        self.update_view_overlays() 

    def update_view_overlays(self):
        if self._selection_graphic and self._selection_graphic.scene() == self.scene:
            self.scene.removeItem(self._selection_graphic)
        self._selection_graphic = None

        mw = self.main_window
        if not (self.selected_point or self.selected_roi): return

        pen = QPen(QColor("red"), 1.8); pen.setCosmetic(True)
        if self.selected_point:
            y, x = self.selected_point
            self._selection_graphic = self.scene.addEllipse(x - 6, y - 6, 12, 12, pen)
        elif self.selected_roi:
            self._selection_graphic = self.scene.addRect(self.selected_roi, pen)

        if mw.n_frames > 0:
            self.frame_label.setText(f"{self.frame_slider.value()+1}/{mw.n_frames}")
        else:
            self.frame_label.setText("N/A")

    def calculate_and_plot(self):
        mw = self.main_window
        if not (self.selected_point or self.selected_roi) or not mw.strain_out or len(mw.EXX) == 0:
            self.plot_widget.set_data({}, meta={'title': "Poisson's Ratio"})
            return

        poisson_ratios = []
        is_roi = (self.selected_roi is not None)

        num_strain_frames = len(mw.EXX)
        if num_strain_frames == 0:
            self.plot_widget.set_data({}, meta={'title': "Poisson's Ratio", 'xlabel': "Frame", 'ylabel': "Strain data not loaded."})
            return


        for i in range(mw.n_frames):
            exx_field, eyy_field, exy_field = mw.EXX[i], mw.EYY[i], mw.EXY[i]
            if is_roi:
                roi = self.selected_roi
                y_start, y_end = int(max(0, roi.top())), int(min(exx_field.shape[0], roi.bottom()))
                x_start, x_end = int(max(0, roi.left())), int(min(exx_field.shape[1], roi.right()))

                if y_start >= y_end or x_start >= x_end:
                    exx, eyy, exy = np.nan, np.nan, np.nan
                else:
                    with np.errstate(invalid='ignore'):
                        exx = np.nanmean(exx_field[y_start:y_end, x_start:x_end])
                        eyy = np.nanmean(eyy_field[y_start:y_end, x_start:x_end])
                        exy = np.nanmean(exy_field[y_start:y_end, x_start:x_end])
            else:
                y, x = self.selected_point
                sampler = get_sampler(self.cb_interp.currentText())
                exx = sampler(exx_field, y, x)
                eyy = sampler(eyy_field, y, x)
                exy = sampler(exy_field, y, x)

            e1, e2 = principal_strains_from_components(exx, eyy, exy, exy_is_engineering=False)
            axial_strain, trans_strain = (e1, e2) if self.cb_axial.currentIndex() == 0 else (e2, e1)

            if abs(axial_strain) < 1e-12 or np.isnan(axial_strain):
                nu = np.nan
            else:
                nu = -trans_strain / axial_strain
            poisson_ratios.append(nu)

        data = {'x': np.arange(1, num_strain_frames + 1), 'Poisson Ratio': np.array(poisson_ratios)}
        meta = {'xlabel': "Frame Number", 'ylabel': "Poisson's Ratio (ν)", 'title': f"Poisson's Ratio ({'ROI' if is_roi else 'Point'})"}
        self.plot_widget.set_data(data, meta)


class AdvancedPlottingTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent

        self.dic_out = None
        self.strain_out = None
        self.image_paths = []
        self.global_minmax = {}
        self.units = "px"
        self.units_per_pixel = 1.0

        H = QHBoxLayout(self)
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True); left_scroll.setFrameShape(QFrame.NoFrame)
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)

        gb_load = QGroupBox("1. Load Data")
        load_layout = QFormLayout(gb_load)
        self.btn_load_dic = QPushButton("Load DIC Output (.bin)"); self.btn_load_dic.clicked.connect(self._load_dic)
        self.btn_load_strain = QPushButton("Load Strain Output (.bin)"); self.btn_load_strain.clicked.connect(self._load_strain)
        self.btn_load_images = QPushButton("Load Image Sequence"); self.btn_load_images.clicked.connect(self._load_images)
        self.image_list_label = QLabel("0 images loaded.")
        load_layout.addRow(self.btn_load_dic)
        load_layout.addRow(self.btn_load_strain)
        load_layout.addRow(self.btn_load_images)
        load_layout.addRow(self.image_list_label)
        left.addWidget(gb_load)

        gb_control = QGroupBox("2. Plot Control")
        control_layout = QFormLayout(gb_control)
        self.cb_field = QComboBox(); self.cb_field.setEnabled(False)
        self.frame_slider = QSlider(Qt.Horizontal); self.frame_slider.setEnabled(False)
        self.frame_label = QLabel("Frame: N/A")
        self.btn_update_plot = QPushButton("Update Plot")
        self.chk_auto_update = QCheckBox("Update automatically"); self.chk_auto_update.setChecked(True)

        control_layout.addRow("Field:", self.cb_field)
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.frame_slider); slider_layout.addWidget(self.frame_label)
        control_layout.addRow("Frame:", slider_layout)
        control_layout.addRow(self.btn_update_plot, self.chk_auto_update)
        left.addWidget(gb_control)

        gb_proc = QGroupBox("3. Processing & Scale")
        proc_layout = QFormLayout(gb_proc)
        self.chk_normalize = QCheckBox("Normalize color scale globally"); self.chk_normalize.setChecked(True)
        self.smooth_slider = QSlider(Qt.Horizontal); self.smooth_slider.setRange(0, 50); self.smooth_slider.setValue(0)
        self.smooth_label = QLabel("Sigma: 0.0")
        self.cb_exyconv = QComboBox(); self.cb_exyconv.addItems(["Tensor εxy", "Engineering γxy"])
        self.le_lower = QLineEdit(); self.le_upper = QLineEdit()
        proc_layout.addRow(self.chk_normalize)
        smooth_layout = QHBoxLayout(); smooth_layout.addWidget(self.smooth_slider); smooth_layout.addWidget(self.smooth_label)
        proc_layout.addRow("Gaussian Smoothing:", smooth_layout)
        proc_layout.addRow("EXY convention:", self.cb_exyconv)
        proc_layout.addRow("Min Color Scale:", self.le_lower)
        proc_layout.addRow("Max Color Scale:", self.le_upper)
        left.addWidget(gb_proc)

        gb_display = QGroupBox("4. Display & Overlays")
        display_layout = QFormLayout(gb_display)
        self.cb_cmap = QComboBox(); self.cb_cmap.addItems(["jet", "viridis", "turbo", "plasma", "inferno", "coolwarm"])
        self.alpha_slider = QSlider(Qt.Horizontal); self.alpha_slider.setRange(0, 100); self.alpha_slider.setValue(100)
        self.alpha_label = QLabel("100%")
        self.chk_colorbar = QCheckBox("Show colorbar"); self.chk_colorbar.setChecked(True)
        self.chk_contour = QCheckBox("Show contour lines"); self.chk_contour.setChecked(False)
        self.chk_scalebar = QCheckBox("Show scalebar"); self.chk_scalebar.setChecked(True)
        self.chk_xyglyph = QCheckBox("Show XY arrows"); self.chk_xyglyph.setChecked(True)
        btn_calibrate = QPushButton("Calibrate Units..."); btn_calibrate.clicked.connect(self._open_calibration)

        display_layout.addRow("Colormap:", self.cb_cmap)
        alpha_layout = QHBoxLayout(); alpha_layout.addWidget(self.alpha_slider); alpha_layout.addWidget(self.alpha_label)
        display_layout.addRow("Overlay Alpha:", alpha_layout)
        display_layout.addRow(self.chk_colorbar, self.chk_contour)
        display_layout.addRow(self.chk_scalebar, self.chk_xyglyph)
        display_layout.addRow(btn_calibrate)
        left.addWidget(gb_display)

        gb_title = QGroupBox("5. Title & Font")
        title_layout = QFormLayout(gb_title)
        self.le_title = QLineEdit()
        self.cb_font = QComboBox()
        self.sp_fontsize = QSpinBox()
        self.sp_fontsize.setRange(6, 48); self.sp_fontsize.setValue(12)

        fonts = QFontDatabase().families()
        self.cb_font.addItems(fonts)
        try: self.cb_font.setCurrentText("Arial")
        except: pass

        title_layout.addRow("Custom Title:", self.le_title)
        title_layout.addRow("Font:", self.cb_font)
        title_layout.addRow("Font Size:", self.sp_fontsize)
        left.addWidget(gb_title)

        gb_export = QGroupBox("6. Export")
        export_layout = QHBoxLayout(gb_export)
        self.btn_export_png = QPushButton("Export PNG"); self.btn_export_png.clicked.connect(self._export_png)
        self.btn_export_gif = QPushButton("Export GIF"); self.btn_export_gif.clicked.connect(self._export_gif)
        export_layout.addWidget(self.btn_export_png); export_layout.addWidget(self.btn_export_gif)
        left.addWidget(gb_export)
        left.addStretch()
        left_scroll.setWidget(left_widget)

        right = QVBoxLayout()
        self.plot_widget = MplPlotWidget(width=7, height=7, dpi=100)
        right.addWidget(self.plot_widget)

        H.addWidget(left_scroll, 1)
        H.addLayout(right, 3)

        self.btn_update_plot.clicked.connect(self._update_plot)
        self.widgets_for_update = [
            self.cb_field, self.chk_normalize, self.cb_cmap, self.chk_colorbar,
            self.chk_contour, self.chk_scalebar, self.chk_xyglyph, self.cb_exyconv,
            self.frame_slider, self.smooth_slider, self.alpha_slider,
            self.le_lower, self.le_upper, self.le_title, self.cb_font, self.sp_fontsize
        ]
        self.chk_auto_update.toggled.connect(self._connect_auto_update)
        self._connect_auto_update(True)

        self.smooth_slider.valueChanged.connect(lambda v: self.smooth_label.setText(f"Sigma: {v/10.0:.1f}"))
        self.alpha_slider.valueChanged.connect(lambda v: self.alpha_label.setText(f"{v}%"))

    def _connect_auto_update(self, checked):
        for w in self.widgets_for_update:
            try:
                if isinstance(w, (QComboBox, QFontComboBox)):
                    if checked: w.currentIndexChanged.connect(self._update_plot)
                    else: w.currentIndexChanged.disconnect(self._update_plot)
                elif isinstance(w, QCheckBox):
                    if checked: w.toggled.connect(self._update_plot)
                    else: w.toggled.disconnect(self._update_plot)
                elif isinstance(w, (QSlider, QSpinBox)):
                    if checked: w.valueChanged.connect(self._update_plot)
                    else: w.valueChanged.disconnect(self._update_plot)
                elif isinstance(w, QLineEdit):
                    if checked: w.editingFinished.connect(self._update_plot)
                    else: w.editingFinished.disconnect(self._update_plot)
            except (TypeError, RuntimeError):
                pass

    def _open_calibration(self):
        img_to_calibrate = None
        if self.image_paths:
            try:
                img_to_calibrate = iio.imread(self.image_paths[0])
                if img_to_calibrate.ndim == 3: 
                    img_to_calibrate = img_to_calibrate[..., 0]  
            except Exception: pass
        elif self.main_window.ref_img is not None:
            img_to_calibrate = self.main_window.ref_img

        if img_to_calibrate is None:
            QMessageBox.warning(self, "Image Missing", "Please load an image sequence or a reference image first.")
            return

        dlg = CalibrateUnitsDialog(img_to_calibrate, self)
        if dlg.exec():
            if dlg.units_per_pixel and dlg.units:
                self.ed_upp.setText(f"{dlg.units_per_pixel:.8f}")
                self.cb_units.setCurrentText(dlg.units)
                self._apply_units()


    def _load_dic(self):
        if ncorr is None: return QMessageBox.critical(self, "ncorr missing", "ncorr module not found.")
        path, _ = QFileDialog.getOpenFileName(self, "Open DIC output .bin", "", "ncorr bin (*.bin)")
        if path:
            try: self.dic_out = ncorr.load_DIC_output(path); self._update_data_and_ui()
            except Exception as e: QMessageBox.critical(self, "Load error", f"{e}")

    def _load_strain(self):
        if ncorr is None: return QMessageBox.critical(self, "ncorr missing", "ncorr module not found.")
        path, _ = QFileDialog.getOpenFileName(self, "Open Strain output .bin", "", "ncorr bin (*.bin)")
        if path:
            try: self.strain_out = ncorr.load_strain_output(path); self._update_data_and_ui()
            except Exception as e: QMessageBox.critical(self, "Load error", f"{e}")

    def _load_images(self):
        if iio is None: return QMessageBox.critical(self, "imageio missing", "imageio module not found.")
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Image Sequence (Ref + Deformed)", "", "Images (*.png *.jpg *.tif)")
        if paths:
            self.image_paths = sorted(paths, key=natural_key)
            self.image_list_label.setText(f"{len(self.image_paths)} images loaded.")
            self._update_data_and_ui()

    def _get_n_frames(self):
        n_dic = len(self.dic_out.disps) if self.dic_out else 0
        n_strain = len(self.strain_out.strains) if self.strain_out else 0
        return max(n_dic, n_strain)

    def _update_data_and_ui(self):
        current_field = self.cb_field.currentText()
        self.cb_field.clear()

        base_fields = []
        calc_fields = []
        if self.dic_out: base_fields.extend(["U", "V"])
        if self.strain_out:
            calc_fields.extend([
                "|E1| (principal)", "E2 (principal)", "Max Shear Strain (γ_max)", "Tresca Strain",
                "Von Mises (plane-strain)", "Strain Energy Density"
            ])
            base_fields.extend(["EXX", "EYY", "EXY"])

        fields = calc_fields + base_fields
        self.cb_field.addItems(fields)

        if current_field in fields:
            self.cb_field.setCurrentText(current_field)
        elif "EXX" in fields:
            self.cb_field.setCurrentText("EXX")
        elif calc_fields:
            self.cb_field.setCurrentIndex(0)

        n_frames = self._get_n_frames()
        if n_frames > 0 and self.image_paths and len(self.image_paths) != n_frames + 1:
            QMessageBox.warning(self, "Frame Mismatch", f"Data frames ({n_frames}) + ref != image files ({len(self.image_paths)}).")

        if n_frames > 0:
            self.frame_slider.setEnabled(True); self.frame_slider.setRange(0, n_frames - 1)
            self.cb_field.setEnabled(True); self._refresh_data_cache(); self._update_plot()
        else:
            self.frame_slider.setEnabled(False); self.cb_field.setEnabled(False)

    def _refresh_data_cache(self):
        self.global_minmax.clear()
        n_frames = self._get_n_frames()
        if n_frames == 0: return
        all_field_names = [self.cb_field.itemText(i) for i in range(self.cb_field.count())]
        strain_fields = [f for f in all_field_names if f not in ["U", "V"]]

        for field_name in strain_fields:
            all_vals = [self._get_field(field_name, i)[0] for i in range(n_frames)]
            all_vals_flat = np.concatenate([v.flatten() for v in all_vals if v is not None and v.size > 0])
            if all_vals_flat.size > 0:
                self.global_minmax[field_name] = (np.nanpercentile(all_vals_flat, 1), np.nanpercentile(all_vals_flat, 99))

    def _get_field(self, kind: str, idx: int):
        if kind in ("U", "V"):
            if not self.dic_out or idx >= len(self.dic_out.disps): return None, None
            disp = self.dic_out.disps[idx]
            data2d = disp.get_u() if kind == "U" else disp.get_v()
            arr = array2d_to_numpy(data2d.get_array())
            mask = array2d_to_numpy(disp.get_roi().get_mask())
            return arr.astype(float), mask.astype(bool)
        elif kind in ("EXX", "EYY", "EXY"):
            if not self.strain_out or idx >= len(self.strain_out.strains): return None, None
            s = self.strain_out.strains[idx]
            if kind == "EYY": data2d = s.get_eyy()
            elif kind == "EXY": data2d = s.get_exy()
            else: data2d = s.get_exx()
            arr = array2d_to_numpy(data2d.get_array())
            mask = array2d_to_numpy(s.get_roi().get_mask())
            return arr.astype(float), mask.astype(bool)

        else:
            if not self.strain_out or idx >= len(self.strain_out.strains): return None, None
            s = self.strain_out.strains[idx]
            exx = array2d_to_numpy(s.get_exx().get_array()).astype(float)
            eyy = array2d_to_numpy(s.get_eyy().get_array()).astype(float)
            exy = array2d_to_numpy(s.get_exy().get_array()).astype(float)
            mask = array2d_to_numpy(s.get_roi().get_mask()).astype(bool)

            exy_is_engineering = (self.cb_exyconv.currentText() == "Engineering γxy")
            exy_t = exy_tensor_from_convention(exy, exy_is_engineering)

            tr = exx + eyy
            rad = np.sqrt(((exx - eyy) * 0.5)**2 + exy_t**2)
            e1 = (tr * 0.5) + rad
            e2 = (tr * 0.5) - rad

            if kind == "|E1| (principal)": return np.abs(e1), mask
            if kind == "E2 (principal)": return e2, mask
            if kind == "Max Shear Strain (γ_max)": return e1 - e2, mask
            if kind == "Tresca Strain": return (e1 - e2) / 2.0, mask
            if kind == "Von Mises (plane-strain)":
                vm = np.sqrt((2.0/9.0) * ((e1 - e2)**2 + e1**2 + e2**2))
                return vm, mask
            if kind == "Strain Energy Density":
                E = self.main_window.tab3.sp_E.value(); nu = self.main_window.tab3.sp_nu.value()
                gamma = 2.0 * exy_t; fac = E / (1.0 - nu**2); sxx = fac * (exx + nu * eyy)
                syy = fac * (nu * exx + eyy); txy = E / (2.0 * (1.0 + nu)) * gamma
                W = 0.5 * (exx * sxx + eyy * syy + gamma * txy)
                return W, mask
            return None, None

    def _update_plot(self, is_export=False, frame_override=None):
        fig = self.plot_widget.fig
        fig.clear()
        ax = fig.add_subplot(111)

        if self.cb_field.count() == 0:
            ax.text(0.5, 0.5, "Load data to begin", ha='center', va='center', transform=ax.transAxes)
            if not is_export: self.plot_widget.canvas.draw()
            return

        kind = self.cb_field.currentText()
        frame_idx = frame_override if frame_override is not None else self.frame_slider.value()

        self.frame_label.setText(f"{frame_idx + 1}/{self._get_n_frames()}")
        a, m = self._get_field(kind, frame_idx)
        if a is None:
            ax.text(0.5, 0.5, "Data not available for this frame", ha='center', va='center', transform=ax.transAxes)
            if not is_export: self.plot_widget.canvas.draw()
            return

        sigma = self.smooth_slider.value() / 10.0
        if sigma > 0 and m.any():
            a_filled = np.nan_to_num(a, nan=np.nanmedian(a[m]))
            a = gaussian_filter(a_filled, sigma=sigma)
        a[~m] = np.nan

        vmin, vmax = None, None
        try:
            manual_vmin = float(self.le_lower.text()); manual_vmax = float(self.le_upper.text())
            if np.isfinite(manual_vmin) and np.isfinite(manual_vmax) and manual_vmin < manual_vmax:
                vmin, vmax = manual_vmin, manual_vmax
        except (ValueError, TypeError): pass

        if vmin is None:
            if self.chk_normalize.isChecked() and kind in self.global_minmax:
                vmin, vmax = self.global_minmax[kind]
            else:
                valid_data = a[np.isfinite(a)]
                if valid_data.size > 1: vmin, vmax = np.nanpercentile(a, 1), np.nanpercentile(a, 99)

        if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            valid_data = a[np.isfinite(a)]
            if valid_data.size > 0:
                median_val = np.nanmedian(a)
                vmin = median_val - 0.1 if median_val == 0 else median_val - 0.1 * abs(median_val)
                vmax = median_val + 0.1 if median_val == 0 else median_val + 0.1 * abs(median_val)
                if vmin == vmax: vmax = vmin + 1.0
            else: vmin, vmax = 0, 1

        img_idx = frame_idx + 1
        if img_idx < len(self.image_paths):
            try:
                bg = iio.imread(self.image_paths[img_idx])
                if bg.ndim == 3: bg = bg[..., 0]
                ax.imshow(bg, cmap='gray', interpolation='nearest', origin='upper')
            except Exception: pass

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = get_cmap(self.cb_cmap.currentText())
        alpha = self.alpha_slider.value() / 100.0
        ax.imshow(a, cmap=cmap, norm=norm, interpolation='bilinear', origin='upper', alpha=alpha)

        title_text = self.le_title.text() or f"{kind} - Frame {frame_idx + 1}"
        font_name = self.cb_font.currentText()
        font_size = self.sp_fontsize.value()
        ax.set_title(title_text, fontsize=font_size, fontname=font_name)
        ax.axis('off')

        if self.chk_colorbar.isChecked():
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=cax)
            cbar.set_label(kind)

        if self.chk_contour.isChecked():
            ax.contour(a, levels=np.linspace(vmin, vmax, 10), colors='k', linewidths=0.7)

        self._draw_scalebar(a.shape, ax); self._draw_xy_arrows(a.shape, ax)
        
        try:
            fig.tight_layout()
        except Exception:
            pass

        if not is_export:
            self.plot_widget.canvas.draw()

        return fig

    def _draw_scalebar(self, shape, ax):
        if not self.chk_scalebar.isChecked() or self.units == 'px': return
        h, w = shape; upp = self.units_per_pixel
        bar_len_units = 10**np.floor(np.log10(w * upp * 0.2))
        bar_len_px = bar_len_units / upp
        fontprops = FontProperties(size=10)
        scalebar = AnchoredSizeBar(ax.transData, bar_len_px, f"{bar_len_units:.0f} {self.units}",
                                   'lower right', pad=0.5, color='white', frameon=False,
                                   size_vertical=max(1, h/100), fontproperties=fontprops)
        ax.add_artist(scalebar)

    def _draw_xy_arrows(self, shape, ax):
        if not self.chk_xyglyph.isChecked(): return
        h, w = shape
        margin = 0.05 * min(h, w)
        arrow_len = 0.1 * min(h, w)
        x0, y0 = margin, margin

        text_effects = [patheffects.withStroke(linewidth=2, foreground='black')]

        ax.arrow(x0, y0, 0, arrow_len, head_width=arrow_len/3, head_length=arrow_len/3, fc='white', ec='black', lw=1.5)
        ax.text(x0, y0 + arrow_len * 1.2, 'Y', color='white', ha='center', va='top', fontsize=10, path_effects=text_effects)
        ax.arrow(x0, y0, arrow_len, 0, head_width=arrow_len/3, head_length=arrow_len/3, fc='white', ec='black', lw=1.5)
        ax.text(x0 + arrow_len * 1.2, y0, 'X', color='white', ha='left', va='center', fontsize=10, path_effects=text_effects)

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "plot.png", "PNG Image (*.png)")
        if path:
            fig = self._update_plot(is_export=True)
            if fig:
                fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
                QMessageBox.information(self, "Success", f"Plot saved to {path}")

    def _export_gif(self):
        if iio is None: return QMessageBox.critical(self, "imageio missing", "Cannot export GIF.")
        n_frames = self._get_n_frames()
        if n_frames <= 1: return QMessageBox.warning(self, "Not an Animation", "Need more than one frame to create a GIF.")

        path, _ = QFileDialog.getSaveFileName(self, "Save Animation", "animation.gif", "GIF Animation (*.gif)")
        if not path: return

        original_frame = self.frame_slider.value()
        progress = QProgressDialog("Exporting GIF...", "Cancel", 0, n_frames * 2, self)
        progress.setWindowModality(Qt.WindowModal)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                png_paths = []

                progress.setLabelText("Step 1/2: Generating frames...")
                for i in range(n_frames):
                    progress.setValue(i)
                    if progress.wasCanceled(): return

                    fig = self._update_plot(is_export=True, frame_override=i)
                    if fig:
                        filepath = temp_path / f"frame_{i:04d}.png"
                        fig.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0.1)
                        png_paths.append(filepath)
                    QApplication.processEvents()

                progress.setLabelText("Step 2/2: Compiling GIF...")
                images = []
                for i, filepath in enumerate(png_paths):
                    progress.setValue(n_frames + i)
                    if progress.wasCanceled(): return
                    images.append(iio.imread(filepath))
                    QApplication.processEvents()

                iio.mimsave(path, images, fps=10)
                progress.setValue(n_frames * 2)
                QMessageBox.information(self, "Success", f"Animation saved to {path}")

        except Exception as e:
            QMessageBox.critical(self, "GIF Export Error", f"An error occurred during export:\n{e}")
        finally:
            self.frame_slider.setValue(original_frame)
            self._update_plot()

    def export_csv(self):
        if self.cb_field.count() == 0:
            QMessageBox.warning(self, "No data", "No data field is loaded/selected.")
            return

        kind = self.cb_field.currentText(); frame_idx = self.frame_slider.value()
        a, m = self._get_field(kind, frame_idx)
        if a is None:
            QMessageBox.warning(self, "No data", "Could not retrieve data for the current frame.")
            return

        a[~m] = np.nan
        default_name = f"{kind.replace(' ', '_').replace('(', '').replace(')', '').replace('|', '')}_frame_{frame_idx+1}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save Data as CSV", default_name, "CSV Files (*.csv)")
        if path:
            try:
                np.savetxt(path, a, delimiter=',')
                QMessageBox.information(self, "Success", f"Data exported to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Could not save file:\n{e}")


class CMODTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.selA: Optional[Tuple[float,float]] = None
        self.selB: Optional[Tuple[float,float]] = None
        self.image_paths = []
        H = QHBoxLayout(self)

        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True); left_scroll.setFrameShape(QFrame.NoFrame)
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)

        gb_load = QGroupBox("1. Load Data")
        gl = QGridLayout(gb_load)
        self.btn_load_dic = QPushButton("Load DIC out .bin"); self.btn_load_dic.clicked.connect(self.main_window.load_dic_bin)
        self.btn_load_force = QPushButton("Load Force Data (.csv)"); self.btn_load_force.clicked.connect(self.main_window.load_force_data)
        self.btn_load_images = QPushButton("Load Image Sequence"); self.btn_load_images.clicked.connect(self._load_images)
        self.image_list_label = QLabel("0 images loaded.")
        gl.addWidget(self.btn_load_dic, 0, 0, 1, 2)
        gl.addWidget(self.btn_load_force, 1, 0, 1, 2)
        gl.addWidget(self.btn_load_images, 2, 0, 1, 2)
        gl.addWidget(self.image_list_label, 3, 0, 1, 2)
        left.addWidget(gb_load)

        gb_select_info = QGroupBox("2. Selections")
        sl = QVBoxLayout(gb_select_info)
        self.info_label = QLabel("Click two points on the image for CMOD.")
        self.btn_clear = QPushButton("Clear Selections"); self.btn_clear.clicked.connect(self.clear_points)
        sl.addWidget(self.info_label); sl.addWidget(self.btn_clear)
        left.addWidget(gb_select_info)

        gb_opts = QGroupBox("3. Options & Calibration")
        ol = QFormLayout(gb_opts)
        self.cb_interp = QComboBox(); self.cb_interp.addItems(["Bilinear (subpixel)", "Nearest (fast)"])
        self.cb_units = QComboBox(); self.cb_units.addItems(["mm", "in", "m", "cm", "px"])
        self.ed_upp = QLineEdit("1.0")
        self.btn_calibrate = QPushButton("Calibrate...")
        self.btn_calibrate.clicked.connect(self.open_calibration)
        ol.addRow("Interpolation:", self.cb_interp)
        ol.addRow("Units:", self.cb_units)
        upp_layout = QHBoxLayout()
        upp_layout.addWidget(self.ed_upp)
        upp_layout.addWidget(self.btn_calibrate)
        ol.addRow("Units per Pixel:", upp_layout)
        left.addWidget(gb_opts)

        gb_plot = QGroupBox("Load vs. CMOD")
        pl = QVBoxLayout(gb_plot)
        self.plot_widget = MplPlotWidget()
        pl.addWidget(self.plot_widget)
        left.addWidget(gb_plot)
        left.addStretch()
        left_scroll.setWidget(left_widget)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.scene = QGraphicsScene(self); self.view = FitGraphicsView(self)
        self.view.setScene(self.scene); self.view.viewport().installEventFilter(self)

        self.frame_slider = QSlider(Qt.Horizontal); self.frame_slider.setEnabled(False)
        self.frame_label = QLabel("Frame: N/A")
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Image Frame:"))
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_label)

        right_layout.addLayout(slider_layout)
        right_layout.addWidget(self.view)

        H.addWidget(left_scroll, 1)
        H.addWidget(right_panel, 2)

        self._pix_item = None; self._crossA = None; self._crossB = None
        self.ed_upp.editingFinished.connect(self._apply_units)
        self.cb_units.currentIndexChanged.connect(self._apply_units)
        self.cb_interp.currentIndexChanged.connect(self.plot_cmod)
        self.frame_slider.valueChanged.connect(self.update_view)
        self._apply_units()

    def _load_images(self):
        if iio is None: return QMessageBox.critical(self, "imageio missing", "imageio module not found.")
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Image Sequence (Ref + Deformed)", "", "Images (*.png *.jpg *.tif)")
        if paths:
            self.image_paths = sorted(paths, key=natural_key)
            self.image_list_label.setText(f"{len(self.image_paths)} images loaded.")

            if self.main_window.ref_img is None:
                self.main_window.load_ref_img(path=self.image_paths[0])

            num_images = len(self.image_paths)
            if num_images > 0:
                self.frame_slider.setRange(0, num_images - 1)
                self.frame_slider.setValue(0)
                self.frame_slider.setEnabled(True)
            else:
                self.frame_slider.setEnabled(False)
            self.update_view()

    def open_calibration(self):
        img_to_calibrate = None
        if self.image_paths:
            try:
                img_to_calibrate = iio.imread(self.image_paths[0])
                if img_to_calibrate.ndim == 3: 
                    img_to_calibrate = img_to_calibrate[..., 0]  
            except Exception: pass
        elif self.main_window.ref_img is not None:
            img_to_calibrate = self.main_window.ref_img

        if img_to_calibrate is None:
            QMessageBox.warning(self, "Image Missing", "Please load an image sequence or a reference image first.")
            return

        dlg = CalibrateUnitsDialog(img_to_calibrate, self)
        if dlg.exec():
            if dlg.units_per_pixel and dlg.units:
                self.ed_upp.setText(f"{dlg.units_per_pixel:.8f}")
                self.cb_units.setCurrentText(dlg.units)
                self._apply_units()


    def _apply_units(self):
        mw = self.main_window
        try:
            mw.units_per_px = float(self.ed_upp.text())
            mw.units = self.cb_units.currentText()
            self.plot_cmod()
        except (ValueError, TypeError): QMessageBox.warning(self, "Invalid Input", "Units per pixel must be a valid number.")

    def eventFilter(self, obj, ev):
        if obj is self.view.viewport() and ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.LeftButton:
            pt = self.view.mapToScene(ev.pos()); self.pick_point((float(pt.y()), float(pt.x()))); return True
        return super().eventFilter(obj, ev)

    def pick_point(self, yx):
        if self.selA is None: self.selA = yx; self.info_label.setText(f"Point A: ({yx[1]:.1f}, {yx[0]:.1f}). Select Point B.")
        elif self.selB is None: self.selB = yx; self.info_label.setText(f"A:({self.selA[1]:.1f}, {self.selA[0]:.1f}), B:({yx[1]:.1f}, {yx[0]:.1f})"); self.plot_cmod()
        else: self.clear_points(); self.selA = yx; self.info_label.setText(f"Point A: ({yx[1]:.1f}, {yx[0]:.1f}). Select Point B.")
        self.update_view()

    def _get_displaced_point(self, yx: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float, bool]]:
        if yx is None:
            return None

        ref_y, ref_x = yx
        mw = self.main_window
        frame_idx = self.frame_slider.value()

        if frame_idx == 0 or not mw.U or not self.image_paths:
            return ref_y, ref_x, True

        disp_idx = frame_idx - 1
        if not (0 <= disp_idx < len(mw.U)):
            return ref_y, ref_x, False

        mask = mw.MASK[disp_idx]
        h, w = mask.shape
        is_valid = False
        if 0 <= ref_y < h and 0 <= ref_x < w:
            is_valid = mask[int(round(ref_y)), int(round(ref_x))]

        sampler = get_sampler(self.cb_interp.currentText())
        u = sampler(mw.U[disp_idx], ref_y, ref_x)
        v = sampler(mw.V[disp_idx], ref_y, ref_x)

        return ref_y + v, ref_x + u, is_valid

    def update_view(self):
        frame_idx = self.frame_slider.value()
        image_to_show = None

        if self.image_paths and 0 <= frame_idx < len(self.image_paths):
            try:
                im = iio.imread(self.image_paths[frame_idx])
                if im.ndim == 3: im = im[..., 0]
                image_to_show = np.clip(im, 0, 255).astype(np.uint8)
                if frame_idx == 0: self.frame_label.setText("Ref")
                else: self.frame_label.setText(f"{frame_idx}/{len(self.image_paths) - 1}")
            except Exception as e:
                self.frame_label.setText("Error")
        elif self.main_window.ref_img is not None:
             image_to_show = self.main_window.ref_img
             self.frame_label.setText("Ref")

        if image_to_show is None: return

        pix = QPixmap.fromImage(np_to_qimage(image_to_show))
        if self._pix_item is None: self._pix_item = self.scene.addPixmap(pix)
        else: self._pix_item.setPixmap(pix)

        self.scene.setSceneRect(QRectF(pix.rect()))
        if self.view._is_fitted_in_view:
             self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        pt_A_info = self._get_displaced_point(self.selA)
        pt_B_info = self._get_displaced_point(self.selB)
        self._draw_cross(pt_A_info, "A")
        self._draw_cross(pt_B_info, "B")

    def _draw_cross(self, point_info: Optional[Tuple[float,float,bool]], tag: str):
        target_cross_attr = "_crossA" if tag == "A" else "_crossB"

        target_cross = getattr(self, target_cross_attr, None)
        if target_cross and target_cross.scene() == self.scene:
            self.scene.removeItem(target_cross)

        new_cross = None
        if point_info:
            y, x, is_valid = point_info
            color = QColor("red") if tag == "A" else QColor("blue")
            if not is_valid:
                color = QColor(128, 128, 128)
            pen = QPen(color, 1.8); pen.setCosmetic(True)
            new_cross = self.scene.addEllipse(x-5, y-5, 10, 10, pen)

        setattr(self, target_cross_attr, new_cross)

    def clear_points(self):
        self.selA, self.selB = None, None; self.info_label.setText("Click two points for CMOD.")
        self.plot_widget.set_data({}, meta={'title': "Load vs. CMOD"})
        self.update_view()

    def plot_cmod(self):
        mw = self.main_window
        mw.cmod_series = None

        if not all([self.selA, self.selB, mw.dic_out, len(mw.U) > 0, mw.load_data is not None]):
            self.plot_widget.set_data({}, meta={'title': "Load vs. CMOD", 'xlabel': "CMOD", 'ylabel': "Load"})
            return

        sampler = get_sampler(self.cb_interp.currentText())

        def get_series(yx):
            U_series, V_series = [], []
            for i in range(mw.n_frames):
                u = sampler(mw.U[i], yx[0], yx[1]); v = sampler(mw.V[i], yx[0], yx[1])
                U_series.append(u); V_series.append(v)
            return np.array(U_series), np.array(V_series)

        uA, vA = get_series(self.selA); uB, vB = get_series(self.selB)

        XA_px = np.array([self.selA[1], self.selA[0]])
        XB_px = np.array([self.selB[1], self.selB[0]])
        AB0_px = XB_px - XA_px
        L0_px = float(np.linalg.norm(AB0_px))
        if L0_px == 0:
            QMessageBox.warning(self, "Points coincide", "Pick two distinct points."); return

        t_hat = AB0_px / L0_px
        n_hat = np.array([-t_hat[1], t_hat[0]])

        cmod_series = []
        for i in range(mw.n_frames):
            xA_px = XA_px + np.array([uA[i], vA[i]])
            xB_px = XB_px + np.array([uB[i], vB[i]])
            d_px = xB_px - xA_px
            cmod_px = float(np.dot(d_px, n_hat))
            cmod_series.append(cmod_px * mw.units_per_px)

        mw.cmod_series = np.array(cmod_series)
        load_frames = mw.load_data[:, 0]
        load_values = mw.load_data[:, 1]

        aligned_load = np.interp(np.arange(1, mw.n_frames + 1), load_frames, load_values, left=np.nan, right=np.nan)

        data = {'x': mw.cmod_series, 'Load': aligned_load}
        meta = {'xlabel': f"CMOD [{mw.units}]", 'ylabel': "Load [N]", 'title': "Load vs. CMOD"}
        self.plot_widget.set_data(data, meta)


class FractureEnergyTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent

        main_layout = QHBoxLayout(self)

        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True); left_scroll.setFrameShape(QFrame.NoFrame)
        left_panel = QFrame()
        left_scroll.setWidget(left_panel)

        left_vbox_layout = QVBoxLayout(left_panel)

        gb_inputs = QGroupBox("1. Specimen Parameters")
        input_layout = QFormLayout(gb_inputs)
        self.mass_input = QDoubleSpinBox(); self.mass_input.setSuffix(" kg"); self.mass_input.setDecimals(4); self.mass_input.setRange(0, 1000)
        self.deflection_input = QDoubleSpinBox(); self.deflection_input.setSuffix(" mm"); self.deflection_input.setDecimals(4); self.deflection_input.setRange(0, 1000)
        self.lig_width_input = QDoubleSpinBox(); self.lig_width_input.setSuffix(" mm"); self.lig_width_input.setDecimals(2); self.lig_width_input.setRange(0, 1000)
        self.lig_height_input = QDoubleSpinBox(); self.lig_height_input.setSuffix(" mm"); self.lig_height_input.setDecimals(2); self.lig_height_input.setRange(0, 1000)

        input_layout.addRow("Specimen Mass (m):", self.mass_input)
        input_layout.addRow("Max Deflection (δ_max):", self.deflection_input)
        input_layout.addRow("Ligament Width (b):", self.lig_width_input)
        input_layout.addRow("Ligament Height (d-a₀):", self.lig_height_input)
        left_vbox_layout.addWidget(gb_inputs)

        self.btn_calculate = QPushButton("2. Calculate Fracture Energy"); self.btn_calculate.clicked.connect(self.calculate_fracture_energy)
        left_vbox_layout.addWidget(self.btn_calculate)

        gb_results = QGroupBox("3. Results")
        results_layout = QFormLayout(gb_results)
        self.work_label = QLabel("N/A"); self.work_label.setStyleSheet("font-weight: bold;")
        self.gravity_work_label = QLabel("N/A"); self.gravity_work_label.setStyleSheet("font-weight: bold;")
        self.area_label = QLabel("N/A"); self.area_label.setStyleSheet("font-weight: bold;")
        self.gf_label = QLabel("N/A"); self.gf_label.setStyleSheet("font-size: 16pt; color: green; font-weight: bold;")

        results_layout.addRow("Work of Fracture (W₀) [J]:", self.work_label)
        results_layout.addRow("Work by Gravity (mgδ_max) [J]:", self.gravity_work_label)
        results_layout.addRow("Ligament Area (A_lig) [m²]:", self.area_label)
        results_layout.addRow("Fracture Energy (Gf) [J/m² or N/m]:", self.gf_label)
        left_vbox_layout.addWidget(gb_results)

        left_vbox_layout.addStretch()

        gb_plot = QGroupBox("Load-CMOD Curve with Work of Fracture (W₀)")
        plot_layout = QVBoxLayout(gb_plot)
        self.plot_widget = MplPlotWidget()
        plot_layout.addWidget(self.plot_widget)

        main_layout.addWidget(left_scroll, 1)
        main_layout.addWidget(gb_plot, 2)

    def calculate_fracture_energy(self):
        mw = self.main_window
        if mw.cmod_series is None or mw.load_data is None:
            QMessageBox.warning(self, "Data Missing", "Please calculate CMOD and load force data in the 'CMOD Calculator' tab first.")
            return

        try:
            mass = self.mass_input.value()
            delta_max = self.deflection_input.value() / 1000.0
            lig_width = self.lig_width_input.value() / 1000.0
            lig_height = self.lig_height_input.value() / 1000.0
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter valid numbers for all parameters.")
            return

        if lig_width <= 0 or lig_height <= 0:
            QMessageBox.critical(self, "Input Error", "Ligament dimensions must be positive.")
            return

        ligament_area = lig_width * lig_height
        self.area_label.setText(f"{ligament_area:.6f}")

        if mw.units != 'mm':
            reply = QMessageBox.question(self, "Unit Mismatch",
                f"CMOD is in [{mw.units}], but fracture energy calculations assume [mm]. Should the CMOD values be converted from {mw.units} to mm?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.No:
                return

        cmod_m = mw.cmod_series / 1000.0

        load_frames = mw.load_data[:, 0]
        load_values_n = mw.load_data[:, 1]

        aligned_load = np.interp(np.arange(1, mw.n_frames + 1), load_frames, load_values_n, left=np.nan, right=np.nan)

        valid_mask = ~np.isnan(cmod_m) & ~np.isnan(aligned_load)
        if not np.any(valid_mask):
            QMessageBox.warning(self, "Calculation Error", "No valid overlapping data for Load and CMOD.")
            return

        cmod_valid = cmod_m[valid_mask]
        load_valid = aligned_load[valid_mask]

        sort_indices = np.argsort(cmod_valid)
        cmod_sorted = cmod_valid[sort_indices]
        load_sorted = load_valid[sort_indices]

        positive_load_mask = load_sorted >= 0

        work_of_fracture_W0 = np.trapz(load_sorted[positive_load_mask], cmod_sorted[positive_load_mask])
        self.work_label.setText(f"{work_of_fracture_W0:.4f}")

        work_gravity = mass * 9.81 * delta_max
        self.gravity_work_label.setText(f"{work_gravity:.4f}")

        fracture_energy_Gf = (work_of_fracture_W0 + work_gravity) / ligament_area
        self.gf_label.setText(f"{fracture_energy_Gf:.2f}")

        data = {'x': cmod_sorted * 1000, 'Load-CMOD Curve': load_sorted}
        meta = {
            'xlabel': "CMOD [mm]",
            'ylabel': "Load [N]",
            'title': f"Load vs. CMOD (W₀ = {work_of_fracture_W0:.4f} J)"
        }
        self.plot_widget.options.show_fill_under = True
        self.plot_widget.options.fill_under_alpha = 0.3
        self.plot_widget.set_data(data, meta)
        self.plot_widget.options.show_fill_under = False


class BinToCsvTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.dic_data = None
        self.strain_data = None

        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        gb_load = QGroupBox("1. Load .bin Files")
        load_layout = QFormLayout(gb_load)
        self.btn_load_dic = QPushButton("Load DIC out .bin")
        self.btn_load_strain = QPushButton("Load Strain out .bin")
        self.dic_path_label = QLabel("<i>Not loaded</i>")
        self.strain_path_label = QLabel("<i>Not loaded</i>")
        load_layout.addRow(self.btn_load_dic, self.dic_path_label)
        load_layout.addRow(self.btn_load_strain, self.strain_path_label)
        main_layout.addWidget(gb_load)

        gb_export = QGroupBox("2. Export Options")
        export_layout = QFormLayout(gb_export)
        self.single_file_radio = QRadioButton("Export all frames to a single CSV file")
        self.per_frame_radio = QRadioButton("Export each frame to a separate CSV file")
        self.single_file_radio.setChecked(True)
        self.btn_export = QPushButton("Export to CSV")
        self.status_label = QLabel("Ready to export.")
        export_layout.addRow(self.single_file_radio)
        export_layout.addRow(self.per_frame_radio)
        export_layout.addRow(self.btn_export)
        export_layout.addRow(QLabel("Status:"), self.status_label)
        main_layout.addWidget(gb_export)

        main_layout.addStretch()

        self.btn_load_dic.clicked.connect(self.load_dic)
        self.btn_load_strain.clicked.connect(self.load_strain)
        self.btn_export.clicked.connect(self.export_csv)

    def load_dic(self):
        if ncorr is None:
            QMessageBox.critical(self, "ncorr missing", "ncorr module not found.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open DIC output .bin", "", "ncorr bin (*.bin)")
        if path:
            try:
                self.dic_data = ncorr.load_DIC_output(path)
                self.dic_path_label.setText(os.path.basename(path))
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Could not load DIC .bin\n{e}")

    def load_strain(self):
        if ncorr is None:
            QMessageBox.critical(self, "ncorr missing", "ncorr module not found.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open Strain output .bin", "", "ncorr bin (*.bin)")
        if path:
            try:
                self.strain_data = ncorr.load_strain_output(path)
                self.strain_path_label.setText(os.path.basename(path))
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Could not load Strain .bin\n{e}")

    def export_csv(self):
        if not self.dic_data and not self.strain_data:
            QMessageBox.warning(self, "No Data", "Please load at least one .bin file first.")
            return

        if self.per_frame_radio.isChecked():
            export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
            if not export_dir: return
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Save Combined CSV", "", "CSV Files (*.csv)")
            if not path: return
            export_dir = os.path.dirname(path)
            export_filename = os.path.basename(path)

        n_frames_dic = len(self.dic_data.disps) if self.dic_data else 0
        n_frames_strain = len(self.strain_data.strains) if self.strain_data else 0
        n_frames = max(n_frames_dic, n_frames_strain)
        if n_frames == 0:
            QMessageBox.warning(self, "No Data", "Loaded files contain no data frames.")
            return

        prog = QProgressDialog("Exporting to CSV...", "Cancel", 0, n_frames, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)

        header = ["frame", "x_px", "y_px"]
        if self.dic_data: header.extend(["u_px", "v_px"])
        if self.strain_data: header.extend(["exx", "eyy", "exy"])

        all_frames_writer = None
        csv_file = None
        if self.single_file_radio.isChecked():
            try:
                csv_file = open(os.path.join(export_dir, export_filename), 'w', newline='')
                all_frames_writer = csv.writer(csv_file)
                all_frames_writer.writerow(header)
            except Exception as e:
                QMessageBox.critical(self, "File Error", f"Could not open file for writing: {e}")
                return

        for i in range(n_frames):
            prog.setValue(i)
            if prog.wasCanceled():
                if csv_file: csv_file.close()
                return

            U, V, MASK_D = (None, None, None)
            if self.dic_data and i < n_frames_dic:
                disp = self.dic_data.disps[i]
                U = array2d_to_numpy(disp.get_u().get_array())
                V = array2d_to_numpy(disp.get_v().get_array())
                MASK_D = array2d_to_numpy(disp.get_roi().get_mask())

            EXX, EYY, EXY, MASK_S = (None, None, None, None)
            if self.strain_data and i < n_frames_strain:
                strain = self.strain_data.strains[i]
                EXX = array2d_to_numpy(strain.get_exx().get_array())
                EYY = array2d_to_numpy(strain.get_eyy().get_array())
                EXY = array2d_to_numpy(strain.get_exy().get_array())
                MASK_S = array2d_to_numpy(strain.get_roi().get_mask())
            
            final_mask = None
            if MASK_D is not None and MASK_S is not None:
                final_mask = MASK_D & MASK_S
            elif MASK_D is not None:
                final_mask = MASK_D
            elif MASK_S is not None:
                final_mask = MASK_S
            else:
                continue

            coords = np.argwhere(final_mask)
            y_coords, x_coords = coords[:, 0], coords[:, 1]
            
            writer = None
            current_csv_file = None
            if self.per_frame_radio.isChecked():
                frame_path = os.path.join(export_dir, f"frame_{i+1:04d}.csv")
                try:
                    current_csv_file = open(frame_path, 'w', newline='')
                    writer = csv.writer(current_csv_file)
                    writer.writerow(header)
                except Exception as e:
                    QMessageBox.warning(self, "File Error", f"Could not write to {frame_path}: {e}")
                    continue
            else:
                writer = all_frames_writer

            for y, x in zip(y_coords, x_coords):
                row = [i + 1, x, y]
                if self.dic_data:
                    row.extend([U[y, x] if U is not None else 'nan', V[y, x] if V is not None else 'nan'])
                if self.strain_data:
                    row.extend([EXX[y, x] if EXX is not None else 'nan', EYY[y, x] if EYY is not None else 'nan', EXY[y, x] if EXY is not None else 'nan'])
                writer.writerow(row)
            
            if current_csv_file:
                current_csv_file.close()

        if csv_file:
            csv_file.close()

        prog.setValue(n_frames)
        self.status_label.setText(f"Successfully exported {n_frames} frames.")
        QMessageBox.information(self, "Success", f"Export completed to {export_dir}")


class PointInspectorApp(QMainWindow):
    ref_img_updated = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Post-Processing Suite")
        g = self.screen().availableGeometry()
        self.setMinimumSize(1400, 900); self.resize(int(g.width()*0.86), int(g.height()*0.86))
        self.move(g.center() - self.rect().center()); self.settings = QSettings("MyCompany", "PointInspector")
        if (geom_bytes := self.settings.value('inspector/geom', type=QByteArray)): self.restoreGeometry(geom_bytes)

        self.global_plot_options = self.load_global_options()

        self.dic_out = None; self.strain_out = None; self.ref_img = None
        self.load_data: Optional[np.ndarray] = None
        self.cmod_series: Optional[np.ndarray] = None
        self.units = "mm"; self.units_per_px = 1.0; self.n_frames = 0
        self.U: List[np.ndarray] = []; self.V: List[np.ndarray] = []; self.MASK: List[np.ndarray] = []
        self.EXX: List[np.ndarray] = []; self.EYY: List[np.ndarray] = []; self.EXY: List[np.ndarray] = []
        self.interp_kind = 'Bilinear (subpixel)'; self.exy_is_engineering = False

        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)
        self.tab1 = PointInspectorTab(self)
        self.tab2 = RelativeDisplacementTab(self)
        self.tab3 = SDICalculatorTab(self)
        self.tab4 = PoissonRatioTab(self)
        self.tab5 = AdvancedPlottingTab(self)
        self.tab6 = CMODTab(self)
        self.tab7 = FractureEnergyTab(self)
        self.tab8 = BinToCsvTab(self)

        self.tabs.addTab(self.tab1, "Point Inspector")
        self.tabs.addTab(self.tab2, "Relative Displacement")
        self.tabs.addTab(self.tab3, "Surface Damage Index (SDI)")
        self.tabs.addTab(self.tab4, "Poisson's Ratio")
        self.tabs.addTab(self.tab5, "Advanced Plotting")
        self.tabs.addTab(self.tab6, "CMOD Calculator")
        self.tabs.addTab(self.tab7, "Fracture Energy (Gf)")
        self.tabs.addTab(self.tab8, ".bin to .csv")

        self.create_menus()

    def create_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("E&xit", self.close, QKeySequence.Quit)
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction("Global Plot Options...", self.edit_global_options)

    def edit_global_options(self):
        dialog = PlotOptionsDialog(self.global_plot_options, self)
        if dialog.exec():
            self.global_plot_options = dialog.get_options()
            self.save_global_options()
            reply = QMessageBox.question(self, "Apply Globally",
                "Apply new global settings to all currently open plots?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.apply_global_options_to_all()

    def apply_global_options_to_all(self):
        for plot_widget in self.findChildren(MplPlotWidget):
            plot_widget.options = dataclasses.replace(self.global_plot_options)
            plot_widget.update_plot()

    def load_global_options(self) -> PlotOptions:
        settings_str = self.settings.value("global_plot_options", "{}")
        try:
            settings_dict = json.loads(settings_str)
            return PlotOptions.from_dict(settings_dict)
        except (json.JSONDecodeError, TypeError):
            return PlotOptions()

    def save_global_options(self):
        settings_dict = self.global_plot_options.to_dict()
        self.settings.setValue("global_plot_options", json.dumps(settings_dict))

    def closeEvent(self, e):
        if hasattr(self, 'settings'):
            self.settings.setValue('inspector/geom', self.saveGeometry())
            self.save_global_options()
        super().closeEvent(e)

    def load_dic_bin(self):
        if ncorr is None: return QMessageBox.critical(self, "ncorr missing", "ncorr module not found.")
        path, _ = QFileDialog.getOpenFileName(self, "Open DIC output .bin", "", "ncorr bin (*.bin)")
        if not path: return
        try:
            self.dic_out = ncorr.load_DIC_output(path)
            self._extract_disps()
            self.statusBar().showMessage(f"DIC loaded: {os.path.basename(path)} — frames={self.n_frames}")
            self.tab1.refresh_plots()
            if self.tab2.selA and self.tab2.selB: self.tab2.plot_relative_distance()
            self.tab5.dic_out = self.dic_out; self.tab5._update_data_and_ui()
            self.tab6.plot_cmod()
        except Exception as e:
            if "allocate" in str(e).lower():
                QMessageBox.critical(self, "Memory Error", "Failed to allocate memory. The DIC file may be too large for the available RAM.")
            else:
                QMessageBox.critical(self, "Load Error", f"Could not load DIC .bin\n{e}")

    def load_strain_bin(self):
        if ncorr is None: return QMessageBox.critical(self, "ncorr missing", "ncorr module not found.")
        path, _ = QFileDialog.getOpenFileName(self, "Open Strain output .bin", "", "ncorr bin (*.bin)")
        if not path: return
        try:
            self.strain_out = ncorr.load_strain_output(path)
            self._extract_strains()
            self.statusBar().showMessage(f"Strain loaded: {os.path.basename(path)}")
            self.tab1.refresh_plots()
            if self.tab3.sdi_values: self.tab3.calculate_sdi()
            self.tab4.calculate_and_plot()
            self.tab5.strain_out = self.strain_out; self.tab5._update_data_and_ui()
        except Exception as e:
            if "allocate" in str(e).lower():
                QMessageBox.critical(self, "Memory Error", "Failed to allocate memory. The Strain file may be too large for the available RAM.")
            else:
                QMessageBox.critical(self, "Load Error", f"Could not load Strain .bin\n{e}")

    def load_ref_img(self, path=None):
        if iio is None: return QMessageBox.critical(self, "imageio missing", "imageio module not found.")
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Open reference image", "", "Images (*.png *.jpg *.tif)")
        if not path: return
        try:
            im = iio.imread(path)
            if im.ndim == 3: im = im[...,0]
            self.ref_img = np.clip(im, 0, 255).astype(np.uint8)
            # Emit the signal instead of manually calling update methods
            self.ref_img_updated.emit()
        except Exception as e:
            QMessageBox.critical(self, "Open error", f"Could not load image\n{e}")

    def load_force_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Force Data", "", "CSV Files (*.csv)")
        if not path: return
        try:
            data = np.loadtxt(path, delimiter=',')
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError("CSV must have exactly two columns: Frame and Force.")
            self.load_data = data
            self.statusBar().showMessage(f"Force data loaded from {os.path.basename(path)}")
            if self.tab6.selA and self.tab6.selB: self.tab6.plot_cmod()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load or parse force data CSV:\n{e}")
            self.load_data = None

    def _extract_disps(self):
        self.U, self.V, self.MASK = [], [], []
        if self.dic_out is None: self.n_frames = 0; return
        disps = self.dic_out.disps; self.n_frames = len(disps)
        for d in disps:
            self.U.append(array2d_to_numpy(d.get_u().get_array()).astype(float))
            self.V.append(array2d_to_numpy(d.get_v().get_array()).astype(float))
            self.MASK.append(array2d_to_numpy(d.get_roi().get_mask()).astype(bool))

    def _extract_strains(self):
        self.EXX, self.EYY, self.EXY = [], [], []
        if self.strain_out is None: return
        strains = self.strain_out.strains; self.n_frames = max(self.n_frames, len(strains))
        for s in strains:
            self.EXX.append(array2d_to_numpy(s.get_exx().get_array()).astype(float))
            self.EYY.append(array2d_to_numpy(s.get_eyy().get_array()).astype(float))
            if hasattr(s, 'get_exy'): self.EXY.append(array2d_to_numpy(s.get_exy().get_array()).astype(float))
            else: self.EXY.append(np.zeros_like(self.EXX[-1]))

    def export_active_csv(self):
        current_tab = self.tabs.currentWidget()
        if hasattr(current_tab, 'export_csv'):
            current_tab.export_csv()
        else:
            QMessageBox.information(self, "Export Not Available", "CSV export is not implemented for the current tab.")
            
def main():
    app = QApplication.instance() or QApplication([])
    w = PointInspectorApp(); w.show(); app.exec()

if __name__ == "__main__":
    with open("dialogs.py", "w") as f:
        f.write("""
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel
class CalibrateUnitsDialog(QDialog):
    def __init__(self, img, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate (Dummy)")
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("This is a placeholder dialog.\\nClosing will return default values."))
        self.units_per_pixel = 1.0
        self.units = "px"
""")
    with open("custom_widgets.py", "w") as f:
        f.write("""
import re
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\\\\d+)', string_)]
""")
    main()