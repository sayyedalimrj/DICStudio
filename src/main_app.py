# main_app.py — workflow shell (updated with DIC/Strain loaders)
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

import sys, os, pickle
import numpy as np
import imageio.v2 as iio

from PySide6.QtCore import Qt, Slot, QSettings
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QGroupBox, QMessageBox, QFrame,
    QFileDialog, QTextEdit, QProgressBar, QInputDialog, QSplitter, QDialog
)

import ncorr

from custom_widgets import np_to_qimage, natural_key, array2d_to_numpy, FitLabel
from dialogs import SetROIsDialog, SetDICParamsDialog, SetSeedsDialog, SetStrainParamsDialog
from analysis_worker import NcorrWorker
from plot_window import ViewPlotsWindow


# New class for displaying the progress popup
class AnalysisProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis in Progress")
        self.setModal(True)
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.message_label = QLabel("Starting analysis...")
        self.message_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(QLabel("Please wait for the analysis to complete."))
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.message_label)
        
        # Prevent the user from closing the dialog
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)

    def set_indeterminate(self, active: bool):
        """Toggles the progress bar between normal and indeterminate mode."""
        if active:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)

    @Slot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    @Slot(str)
    def update_message(self, text):
        self.message_label.setText(text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ncorr Python Studio")
        
        g = self.screen().availableGeometry()
        self.setMinimumSize(900, 600)
        self.resize(int(g.width()*0.86), int(g.height()*0.86))
        self.move(g.center() - self.rect().center())
        self.settings = QSettings()
        if (b := self.settings.value('processing/geom')): self.restoreGeometry(b)
        self._splitter_state = self.settings.value('processing/splitter')

        # State
        self.ref_path, self.cur_paths = "", []
        self.ref_img, self.cur_img_idx = None, -1
        self.roi_mask, self.dic_params, self.strain_params, self.seeds = None, None, None, []
        self.dic_results, self.strain_results, self.worker = None, None, None
        self.plot_windows = []

        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        control_panel = self._build_control_panel()
        display_panel = self._build_display_panel()
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(display_panel, 4)

        self.statusBar().showMessage("Ready.")
        self.update_all_states()

    def load_state_from_auto(self, state_data: dict):
        """Populates the entire application state from the automatic workflow results."""
        self._reset_downstream_data("ref")  # Clear any previous state

        # Set all state variables from the provided dictionary
        self.ref_path = state_data.get("ref_path")
        self.cur_paths = state_data.get("cur_paths", [])
        self.ref_img = state_data.get("ref_img")
        self.roi_mask = state_data.get("roi_mask")
        self.dic_params = state_data.get("dic_params")
        self.strain_params = state_data.get("strain_params")
        self.seeds = state_data.get("seeds", [])
        self.dic_results = state_data.get("dic_results")
        self.strain_results = state_data.get("strain_results")

        self.cur_img_idx = 0 if self.cur_paths else -1
        
        self.log_box.append("--- State loaded from automatic workflow ---")
        
        # Refresh the UI to show the new state
        self.update_all_states()
        self._update_image_display()

    def _build_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # State table
        state_group = QGroupBox("Program State")
        state_layout = QGridLayout(state_group)
        self.state_labels = {}
        states = ["Reference Image", "Current Image(s)", "Region of Interest", "DIC Parameters",
                  "Seeds", "DIC Analysis", "Strain Params", "Strains"]
        for i, s in enumerate(states):
            state_layout.addWidget(QLabel(s), i, 0)
            lab = QLabel("NOT SET"); lab.setAlignment(Qt.AlignRight)
            self.state_labels[s] = lab
            state_layout.addWidget(lab, i, 1)
        layout.addWidget(state_group)

        # Workflow
        ops_group = QGroupBox("Workflow")
        ops = QVBoxLayout(ops_group)
        self.btn_load_ref = QPushButton("1. Load Reference Image")
        self.btn_load_cur = QPushButton("2. Load Current Image(s)")
        self.btn_set_roi = QPushButton("3. Set Region of Interest")
        self.btn_set_params = QPushButton("4. Set DIC Parameters")
        self.btn_set_seeds = QPushButton("5. Set Seeds")
        self.btn_run_analysis = QPushButton("6. Run DIC Analysis")
        self.btn_calc_strain = QPushButton("7. Calculate Strains")
        self.btn_view_plots = QPushButton("8. View Plots")
        # NEW: loaders
        self.btn_load_dic = QPushButton("Load DIC (.bin)")
        self.btn_load_strain = QPushButton("Load Strain (.bin)")
        # Project save/load
        self.btn_save_project = QPushButton("Save Project")
        self.btn_load_project = QPushButton("Load Project")

        for b in [self.btn_load_ref, self.btn_load_cur, self.btn_set_roi, self.btn_set_params,
                  self.btn_set_seeds, self.btn_run_analysis, self.btn_calc_strain, self.btn_view_plots,
                  self.btn_load_dic, self.btn_load_strain]:
            ops.addWidget(b)
        ops.addStretch()
        ops.addWidget(self.btn_save_project)
        ops.addWidget(self.btn_load_project)
        layout.addWidget(ops_group)

        # Log
        log_group = QGroupBox("Log")
        v = QVBoxLayout(log_group)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        v.addWidget(self.log_box)
        layout.addWidget(log_group, 1)

        # Signals
        self.btn_load_ref.clicked.connect(self._trigger_load_reference)
        self.btn_load_cur.clicked.connect(self._trigger_load_current)
        self.btn_set_roi.clicked.connect(self.open_roi_dialog)
        self.btn_set_params.clicked.connect(self.open_dic_params_dialog)
        self.btn_set_seeds.clicked.connect(self.open_set_seeds_dialog)
        self.btn_run_analysis.clicked.connect(self.run_analysis)
        self.btn_calc_strain.clicked.connect(self.open_strain_params_dialog)
        self.btn_view_plots.clicked.connect(self.open_view_plots)
        self.btn_save_project.clicked.connect(self.save_project)
        self.btn_load_project.clicked.connect(self.load_project)
        self.btn_load_dic.clicked.connect(self._load_dic_bins)
        self.btn_load_strain.clicked.connect(self._load_strain_bins)

        return panel

    def _build_display_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.row_splitter = QSplitter(Qt.Horizontal)
        
        self.ref_view = self._create_image_view("Reference Image")
        self.cur_view = self._create_image_view("Current Image(s)")
        self.row_splitter.addWidget(self.ref_view['group'])
        self.row_splitter.addWidget(self.cur_view['group'])
        self.row_splitter.setChildrenCollapsible(False)
        self.row_splitter.setHandleWidth(8)
        self.row_splitter.setStretchFactor(0, 1)
        self.row_splitter.setStretchFactor(1, 1)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("<< Previous")
        self.btn_next = QPushButton("Next >>")
        self.btn_prev.clicked.connect(self._nav_prev)
        self.btn_next.clicked.connect(self._nav_next)
        nav.addStretch(); nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next); nav.addStretch()

        layout.addWidget(self.row_splitter, 1)
        layout.addLayout(nav)
        
        if self._splitter_state:
            try: self.row_splitter.restoreState(self._splitter_state)
            except Exception: pass
            
        return panel

    def _create_image_view(self, title):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        label = FitLabel("Load an image to display.")
        label.setAlignment(Qt.AlignCenter)
        label.setFrameShape(QFrame.StyledPanel)
        label.setMinimumSize(0, 0)
        info = QLabel("Name:\nResolution:")
        layout.addWidget(label, 1); layout.addWidget(info)
        return {"group": group, "label": label, "info": info}

    # ---------- UI state ----------
    def update_all_states(self):
        self._state("Reference Image", self.ref_img is not None)
        self._state("Current Image(s)", bool(self.cur_paths))
        self._state("Region of Interest", self.roi_mask is not None)
        self._state("DIC Parameters", self.dic_params is not None)
        self._state("Seeds", bool(self.seeds))
        self._state("DIC Analysis", self.dic_results is not None)
        self._state("Strain Params", self.strain_params is not None)
        self._state("Strains", self.strain_results is not None)
        self.update_button_states()

    def _state(self, name, ok):
        lab = self.state_labels[name]
        lab.setText("SET" if ok else "NOT SET")
        lab.setStyleSheet(f"color: {'green' if ok else 'red'}; font-weight: bold;")

    def update_button_states(self):
        ref, cur, roi, params, seeds, dic_ok = (
            self.ref_img is not None, bool(self.cur_paths), self.roi_mask is not None,
            self.dic_params is not None, bool(self.seeds), self.dic_results is not None
        )
        self.btn_load_cur.setEnabled(ref)
        self.btn_set_roi.setEnabled(ref)
        self.btn_set_params.setEnabled(roi)
        self.btn_set_seeds.setEnabled(params and roi)
        self.btn_run_analysis.setEnabled(params and seeds and cur)
        self.btn_calc_strain.setEnabled(dic_ok)
        # View Plots only after strain is available
        self.btn_view_plots.setEnabled(self.strain_results is not None)
        self.btn_save_project.setEnabled(ref)
        self.btn_load_project.setEnabled(True)
        self.btn_prev.setEnabled(cur and self.cur_img_idx > 0)
        self.btn_next.setEnabled(cur and self.cur_img_idx < len(self.cur_paths) - 1)

    def _reset_downstream_data(self, from_step):
        steps = ["ref", "cur", "roi", "params", "seeds", "analysis", "strain_params", "strain_results"]
        if from_step in steps:
            k = steps.index(from_step)
            if k <= 1: self.cur_paths, self.cur_img_idx = [], -1
            if k <= 2: self.roi_mask = None
            if k <= 3: self.dic_params = None
            if k <= 4: self.seeds = []
            if k <= 5: self.dic_results = None
            if k <= 6: self.strain_params = None
            if k <= 7: self.strain_results = None
        self._update_image_display(); self.update_all_states()

    # ---------- Display ----------
    def _update_image_display(self):
        if self.ref_img is not None:
            g = self.ref_img.astype(np.uint8)
            h, w = g.shape
            # ROI overlay (green)
            if self.roi_mask is not None:
                overlay = np.dstack([g, g, g])
                mask = self.roi_mask
                overlay[mask, 1] = 255
                overlay = overlay.astype(np.uint8)
                q = QImage(overlay.data, w, h, 3*w, QImage.Format_BGR888).copy()
            else:
                q = np_to_qimage(g)
            pm = QPixmap.fromImage(q)
            self.ref_view['label'].setPixmap(pm.scaled(self.ref_view['label'].size(),
                                                       Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.ref_view['info'].setText(f"Name: {os.path.basename(self.ref_path)}\nResolution: {w} x {h}")
        else:
            self.ref_view['label'].setText("Load Reference Image")
            self.ref_view['info'].setText("Name:\nResolution:")

        if self.cur_paths and self.cur_img_idx != -1:
            cur = iio.imread(self.cur_paths[self.cur_img_idx])
            if cur.ndim == 3: cur = cur[..., 0]
            pm = QPixmap.fromImage(np_to_qimage(cur.astype(np.uint8)))
            self.cur_view['label'].setPixmap(pm.scaled(self.cur_view['label'].size(),
                                                       Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.cur_view['info'].setText(
                f"Name: {os.path.basename(self.cur_paths[self.cur_img_idx])} ({self.cur_img_idx+1}/{len(self.cur_paths)})\n"
                f"Resolution: {cur.shape[1]} x {cur.shape[0]}"
            )
        else:
            self.cur_view['label'].setText("Load Current Image(s)")
            self.cur_view['info'].setText("Name:\nResolution:")

        self.update_button_states()

    # ---------- Workflow handlers ----------
    def _trigger_load_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference Image", "", "Image Files (*.png *.tif *.jpg *.bmp)")
        if path:
            self.load_reference(path)

    def _trigger_load_current(self):
        """Opens a dialog to select multiple 'current' images and loads them."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Current Image(s)",
            "",
            "Image Files (*.png *.tif *.jpg *.bmp)"
        )
        
        if not paths:
            self.log_box.append("Current image selection cancelled.")
            return

        self._reset_downstream_data("cur")
        # Sort the paths naturally to ensure correct sequence (e.g., img10 after img9)
        self.cur_paths = sorted(paths, key=natural_key)
        self.cur_img_idx = 0 if self.cur_paths else -1
        
        self.log_box.append(f"Loaded {len(self.cur_paths)} current image(s).")
        self.update_all_states()
        self._update_image_display()

    def load_reference(self, path):
        if not path: return
        self._reset_downstream_data("ref")
        self.ref_path = path
        try:
            img = iio.imread(self.ref_path)
            self.ref_img = img[..., 0] if img.ndim == 3 else img
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load reference image.\n{e}")
        self.update_all_states(); self._update_image_display()

    def set_roi_from_path(self, path):
        if not path: return
        self._reset_downstream_data("roi")
        try:
            mask = iio.imread(path)
            self.roi_mask = mask > 0 # Ensure boolean mask
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load ROI mask.\n{e}")
        self.update_all_states(); self._update_image_display()

    def set_dic_parameters(self, params_dict):
        self._reset_downstream_data("params")
        self.dic_params = params_dict
        self.update_all_states()

    def set_seeds_automatically(self):
        if not self.dic_params or self.roi_mask is None: return
        self._reset_downstream_data("seeds")
        ys, xs = np.where(self.roi_mask)
        if not len(xs): return # ROI is empty
        num_threads = self.dic_params.get('threads', 1)
        self.seeds = []
        for _ in range(num_threads):
            idx = np.random.randint(0, len(xs))
            self.seeds.append([int(xs[idx]), int(ys[idx])])
        self.update_all_states()

    def open_roi_dialog(self):
        if self.ref_img is None: return
        self._reset_downstream_data("roi")
        d = SetROIsDialog(self, ref_img=self.ref_img)
        if d.exec():
            m = d.get_roi_mask()
            self.roi_mask = m if (m is not None and m.any()) else None
        self.update_all_states(); self._update_image_display()

    def open_dic_params_dialog(self):
        if self.roi_mask is None: return
        self._reset_downstream_data("params")
        d = SetDICParamsDialog(self, self.ref_img, self.roi_mask)
        if d.exec(): self.dic_params = d.get_params()
        self.update_all_states()

    def open_set_seeds_dialog(self):
        if not self.dic_params: return
        self._reset_downstream_data("seeds")
        threads = self.dic_params.get('threads', 1)
        d = SetSeedsDialog(self, self.ref_img, self.roi_mask, threads)
        if d.exec(): self.seeds = d.get_seeds()
        else: self.seeds = []
        self.update_all_states(); self._update_image_display()

    def run_analysis(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "An analysis is already in progress.")
            return
        
        self.log_box.clear()
        self._reset_downstream_data("analysis")
        
        cfg = self.build_cfg()
        all_paths = [self.ref_path] + self.cur_paths
        
        # Create worker and progress dialog
        self.worker = NcorrWorker(cfg, all_paths, self.roi_mask, seeds=self.seeds)
        progress_dialog = AnalysisProgressDialog(self)

        # Connect signals
        self.worker.log.connect(self.log_box.append) # Full log in the background
        self.worker.log.connect(progress_dialog.update_message) # Key messages in the dialog
        self.worker.progress.connect(progress_dialog.update_progress)
        self.worker.done.connect(self._on_worker_done)
        self.worker.finished.connect(progress_dialog.accept) # Close the dialog when the work is finished

        self._set_enabled(False)
        self.worker.start()
        self.statusBar().showMessage("Analysis in progress...")
        
        # Show the dialog
        progress_dialog.exec()

    def build_cfg(self):
        cfg = (self.dic_params or {}).copy()
        if self.strain_params: cfg.update(self.strain_params)

        # Use the output directory from dic_params, or a default if not set
        output_directory = self.dic_params.get('output_dir', '').strip()
        cfg['out_dir'] = output_directory or "./ncorr_output"

        cfg['scale_factor'] = 1
        # required enum names from your bindings
        cfg.setdefault('interp', 'QUINTIC_BSPLINE_PRECOMPUTE')
        cfg.setdefault('subregion', 'CIRCLE')
        cfg.setdefault('dic_config', 'KEEP_MOST_POINTS')
        cfg.setdefault('radius', 20)
        cfg.setdefault('threads', max(1, os.cpu_count() or 1))
        return cfg

    @Slot(bool, str, object, object)
    def _on_worker_done(self, ok, msg, dic_out, strain_out):
        self._set_enabled(True)
        if ok:
            self.dic_results = dic_out
            self.strain_results = strain_out  # may be None if you skipped strain
            self.statusBar().showMessage("Analysis finished.")
            self.log_box.append("\n--- Analysis Finished ---")
        else:
            self.dic_results = None; self.strain_results = None
            self.statusBar().showMessage(f"Analysis failed: {msg}")
            QMessageBox.critical(self, "Analysis Error", msg)
        self.update_all_states()

    def open_strain_params_dialog(self):
        if self.dic_results is None:
            QMessageBox.information(self, "Info", "Run or load a DIC result first.")
            return

        self._reset_downstream_data("strain_params")

        radius, ok = QInputDialog.getInt(self,
                                        "Set Strain Radius",
                                        "Enter the strain radius:",
                                        15,
                                        1,
                                        200,
                                        1)

        if ok:
            self.strain_params = {'strain_radius': radius}
            self._calc_strain_now()
        else:
            self.strain_params = None

        self.update_all_states()

    def _calc_strain_now(self):
        if not self.strain_params or not self.dic_params:
            QMessageBox.critical(self, "Error", "Cannot calculate strain without DIC and strain parameters.")
            return
        try:
            subreg = getattr(ncorr.SUBREGION, self.dic_params.get('subregion', 'CIRCLE'))
            S_in = ncorr.strain_analysis_input(self._mk_dic_input_stub(), self.dic_results,
                                               subreg, int(self.strain_params['strain_radius']))
            S_out = ncorr.strain_analysis(S_in)
            self.strain_results = S_out

            # Immediately save strain_input.bin / strain_output.bin as well
            try:
                out_dir = (self.dic_params or {}).get('output_dir', '').strip() or "./ncorr_output"
                os.makedirs(out_dir, exist_ok=True)
                # reuse the same S_in you built above, or rebuild with current settings:
                S_in_save = S_in if 'S_in' in locals() else ncorr.strain_analysis_input(
                    self._mk_dic_input_stub(), self.dic_results,
                    getattr(ncorr.SUBREGION, self.dic_params.get('subregion', 'CIRCLE')),
                    int(self.strain_params['strain_radius'])
                )
                if hasattr(ncorr, "save_strain_pair"):
                    ncorr.save_strain_pair(
                        os.path.join(out_dir, "strain_input.bin"),
                        os.path.join(out_dir, "strain_output.bin"),
                        S_in_save, self.strain_results
                    )
                else:
                    if hasattr(ncorr, "save_strain_input"):
                        ncorr.save_strain_input(S_in_save, os.path.join(out_dir, "strain_input.bin"))
                    if hasattr(ncorr, "save_strain_output"):
                        ncorr.save_strain_output(self.strain_results, os.path.join(out_dir, "strain_output.bin"))
                self.log_box.append(f"Saved strain binaries to {out_dir}")
            except Exception as _e:
                self.log_box.append(f"Save warning: {_e}")

            self.statusBar().showMessage("Strain calculation completed.")
            self.log_box.append("Strain calculation completed.")
        except Exception as e:
            self.strain_results = None
            QMessageBox.critical(self, "Strain Error", str(e))
        self.update_all_states()

    def _mk_dic_input_stub(self):
        """
        Build a minimal DIC_analysis_input using current images/ROI to pair with already computed DIC_output.
        """
        roi = ncorr.ROI2D(self.roi_mask, 0)
        interp = getattr(ncorr.INTERP, self.dic_params.get('interp', 'QUINTIC_BSPLINE_PRECOMPUTE'))
        subreg = getattr(ncorr.SUBREGION, self.dic_params.get('subregion', 'CIRCLE'))
        cfg = getattr(ncorr.DIC_analysis_config, self.dic_params.get('dic_config', 'KEEP_MOST_POINTS'))
        imgs = [self.ref_path] + self.cur_paths
        return ncorr.DIC_analysis_input(imgs, roi, 1, interp, subreg,
                                        int(self.dic_params.get('radius', 20)),
                                        int(self.dic_params.get('threads', 1)),
                                        cfg, False)

    def open_view_plots(self):
        if not (self.dic_results and self.strain_results):
            QMessageBox.information(self, "Info", "Strain must be available to view plots.")
            return
        units = (self.dic_params or {}).get('units', '')
        upp = float((self.dic_params or {}).get('units_per_pixel', 0.0) or 0.0)
        win = ViewPlotsWindow(self, dic_out=self.dic_results, strain_out=self.strain_results,
                              ref_image=self.ref_img, image_paths=self.cur_paths,
                              units=units, units_per_pixel=upp)
        self.plot_windows.append(win)
        win.show()

    # ---------- Loaders ----------
    def _load_dic_bins(self):
        # Choose pair or individual
        opt = QMessageBox.question(self, "Load DIC", "Load a pair (input+output) ?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
        try:
            if opt == QMessageBox.StandardButton.Yes:
                f_in, _ = QFileDialog.getOpenFileName(self, "Select DIC_input.bin", "", "Binary (*.bin)")
                if not f_in: return
                f_out, _ = QFileDialog.getOpenFileName(self, "Select DIC_output.bin", "", "Binary (*.bin)")
                if not f_out: return
                dic_in, dic_out = ncorr.load_DIC_pair(f_in, f_out)
            else:
                f_out, _ = QFileDialog.getOpenFileName(self, "Select DIC_output.bin", "", "Binary (*.bin)")
                if not f_out: return
                dic_out = ncorr.load_DIC_output(f_out)
                # fabricate minimal DIC_input from current session to support later strain
                if not all([self.roi_mask is not None, self.dic_params is not None, self.ref_path, self.cur_paths]):
                     QMessageBox.warning(self, "Incomplete State", "To calculate strain later, please ensure a reference image, current images, ROI, and DIC parameters are set before loading a standalone DIC output.")
                dic_in = self._mk_dic_input_stub()

            self.dic_results = dic_out
            # Set ROI from first displacement
            if len(dic_out.disps) > 0:
                roi = dic_out.disps[0].get_roi().get_mask()
                self.roi_mask = array2d_to_numpy(roi)
            # If we have no reference image, try to read it from DIC_input
            if self.ref_img is None and hasattr(dic_in, "imgs") and len(dic_in.imgs) > 0:
                try:
                    refA = dic_in.imgs[0].get_gs()
                    self.ref_img = array2d_to_numpy(refA)
                    self.ref_path = "(loaded from bin)"
                except Exception:
                    pass
            self.statusBar().showMessage("DIC loaded from .bin")
            self.update_all_states(); self._update_image_display()
        except Exception as e:
            QMessageBox.critical(self, "Load DIC Error", str(e))

    def _load_strain_bins(self):
        opt = QMessageBox.question(self, "Load Strain", "Load a pair (input+output) ?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
        try:
            if opt == QMessageBox.StandardButton.Yes:
                f_in, _ = QFileDialog.getOpenFileName(self, "Select strain_input.bin", "", "Binary (*.bin)")
                if not f_in: return
                f_out, _ = QFileDialog.getOpenFileName(self, "Select strain_output.bin", "", "Binary (*.bin)")
                if not f_out: return
                s_in, s_out = ncorr.load_strain_pair(f_in, f_out)
                self.strain_results = s_out
                # if DIC_output is missing, pull from s_in
                if self.dic_results is None and hasattr(s_in, "DIC_output"):
                    self.dic_results = s_in.DIC_output
            else:
                f_out, _ = QFileDialog.getOpenFileName(self, "Select strain_output.bin", "", "Binary (*.bin)")
                if not f_out: return
                self.strain_results = ncorr.load_strain_output(f_out)
            # Enable View Plots now
            self.statusBar().showMessage("Strain loaded from .bin")
            self.update_all_states()
        except Exception as e:
            QMessageBox.critical(self, "Load Strain Error", str(e))

    # ---------- Misc ----------
    def save_project(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save Project File", "", "Ncorr Python Project (*.ncorrpy)")
        if not p: return
        state = dict(
            ref_path=self.ref_path, cur_paths=self.cur_paths, roi_mask=self.roi_mask,
            dic_params=self.dic_params, strain_params=self.strain_params, seeds=self.seeds
        )
        try:
            with open(p, "wb") as f: pickle.dump(state, f)
            self.statusBar().showMessage(f"Project saved to {os.path.basename(p)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save project.\n{e}")

    def load_project(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load Project File", "", "Ncorr Python Project (*.ncorrpy)")
        if not p: return
        try:
            with open(p, "rb") as f: state = pickle.load(f)
            self._reset_downstream_data("ref")
            for k, v in state.items(): setattr(self, k, v)
            if self.ref_path and os.path.exists(self.ref_path):
                img = iio.imread(self.ref_path)
                self.ref_img = img[..., 0] if img.ndim == 3 else img
                self.cur_img_idx = 0 if self.cur_paths else -1
            self.statusBar().showMessage(f"Project loaded from {os.path.basename(p)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load project file.\n{e}")
        self.update_all_states(); self._update_image_display()

    def _set_enabled(self, en: bool):
        btns = [self.btn_load_ref, self.btn_load_cur, self.btn_set_roi, self.btn_set_params,
                self.btn_set_seeds, self.btn_run_analysis, self.btn_calc_strain, self.btn_view_plots,
                self.btn_save_project, self.btn_load_project, self.btn_load_dic, self.btn_load_strain,
                self.btn_prev, self.btn_next]
        for b in btns: b.setEnabled(en)
        if en: self.update_button_states()

    def _nav_prev(self):
        if self.cur_img_idx > 0: self.cur_img_idx -= 1; self._update_image_display()

    def _nav_next(self):
        if self.cur_img_idx < len(self.cur_paths) - 1: self.cur_img_idx += 1; self._update_image_display()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # «breakpoint»: low width -> vertical
        if hasattr(self, 'row_splitter'):
            if self.width() < 1100 and self.row_splitter.orientation() != Qt.Vertical:
                self.row_splitter.setOrientation(Qt.Vertical)
            elif self.width() >= 1100 and self.row_splitter.orientation() != Qt.Horizontal:
                self.row_splitter.setOrientation(Qt.Horizontal)
        self._update_image_display()

    def closeEvent(self, ev):
        if hasattr(self, 'settings'):
            self.settings.setValue('processing/geom', self.saveGeometry())
            if hasattr(self, 'row_splitter'):
                self.settings.setValue('processing/splitter', self.row_splitter.saveState())
        for w in self.plot_windows: w.close()
        if self.worker and self.worker.isRunning():
            self.worker.quit(); self.worker.wait()
        ev.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())