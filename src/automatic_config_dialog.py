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
# src/automatic_config_dialog.py (Updated)
import os
import cv2
import imageio.v2 as iio
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QFileDialog,
    QGroupBox, QDialogButtonBox, QHBoxLayout, QLabel, QComboBox, QRadioButton,
    QButtonGroup, QWidget, QApplication, QScrollArea, QFrame, QStackedWidget,
    QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

# Import the dedicated calibration dialog
from dialogs import CalibrateUnitsDialog
from Preprocessing import NcorrConfigTab, pil_to_qpixmap, safe_float, safe_int
from custom_widgets import FitLabel, natural_key
from PIL import Image

class AutomaticConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Automatic Workflow Configuration")
        self.setMinimumWidth(650)
        self.setMinimumHeight(800)
        # --- State Variables ---
        self.config_data = {}
        self.calibration_frame_paths = []
        self.image_file_paths = []
        self.current_frame_index = 0
        try:
            mw = parent if parent is not None else QApplication.activeWindow()
            self._config_generator = NcorrConfigTab(mw)
        except TypeError:
            self._config_generator = NcorrConfigTab()

        # --- Main Dialog Layout ---
        main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # === STEP 1: INPUTS ===
        step1_group = QGroupBox("Step 1: Select Inputs and Analysis Type")
        form1_layout = QFormLayout(step1_group)
        source_group = QGroupBox("Input Source")
        source_layout = QHBoxLayout(source_group)
        self.rb_video = QRadioButton("Video File")
        self.rb_files = QRadioButton("Image Files")
        self.rb_video.setChecked(True)
        source_layout.addWidget(self.rb_video)
        source_layout.addWidget(self.rb_files)
        form1_layout.addRow(source_group)
        self.source_stack = QStackedWidget()
        video_widget = QWidget()
        video_form = QFormLayout(video_widget)
        video_form.setContentsMargins(0,0,0,0)
        self.video_path_le = QLineEdit()
        btn_browse_video = QPushButton("Browse...")
        video_form.addRow("Input Video File:", self.video_path_le)
        video_form.addRow("", btn_browse_video)
        self.source_stack.addWidget(video_widget)
        files_widget = QWidget()
        files_form = QFormLayout(files_widget)
        files_form.setContentsMargins(0,0,0,0)
        self.files_path_label = QLabel("No files selected.")
        btn_browse_files = QPushButton("Browse...")
        files_form.addRow(self.files_path_label)
        files_form.addRow("", btn_browse_files)
        self.source_stack.addWidget(files_widget)
        form1_layout.addRow(self.source_stack)
        self.output_dir_le = QLineEdit()
        btn_browse_output = QPushButton("Browse...")
        form1_layout.addRow("Output Directory:", self.output_dir_le)
        form1_layout.addRow("", btn_browse_output)
        self.extraction_group = QGroupBox("Frame Extraction Method")
        extraction_layout = QVBoxLayout(self.extraction_group)
        self.rb_interval = QRadioButton("By time interval (seconds)")
        self.rb_specific = QRadioButton("Specific frames (e.g., 10, 50, 100)")
        self.rb_range = QRadioButton("Frame range (start, end, step)")
        self.rb_interval.setChecked(True)
        extraction_layout.addWidget(self.rb_interval)
        extraction_layout.addWidget(self.rb_specific)
        extraction_layout.addWidget(self.rb_range)
        self.extractor_options_widget = QWidget()
        self.extractor_options_layout = QFormLayout(self.extractor_options_widget)
        self.extractor_options_layout.setContentsMargins(20, 5, 5, 5)
        extraction_layout.addWidget(self.extractor_options_widget)
        form1_layout.addRow(self.extraction_group)
        self.analysis_mode_combo = QComboBox()
        self.analysis_mode_combo.addItems(["Fixed Reference (for low strain)", "Updating Reference (for high strain)"])
        form1_layout.addRow("Analysis Mode:", self.analysis_mode_combo)
        layout.addWidget(step1_group)

        # === STEP 2: CALIBRATION ===
        self.step2_group = QGroupBox("Step 2: Calibrate Using a Sample Frame")
        step2_layout = QVBoxLayout(self.step2_group)
        nav_layout = QHBoxLayout()
        self.btn_prev_frame = QPushButton("<< Previous Frame")
        self.btn_next_frame = QPushButton("Next Frame >>")
        self.lbl_frame_info = QLabel("Select an input to load frames")
        self.lbl_frame_info.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.btn_prev_frame)
        nav_layout.addWidget(self.lbl_frame_info, 1)
        nav_layout.addWidget(self.btn_next_frame)
        calib_widget = QWidget()
        calib_layout = QHBoxLayout(calib_widget)
        self.preview_label = FitLabel("Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #212121; border-radius: 4px;")
        calib_tools_widget = QWidget()
        calib_tools_layout = QFormLayout(calib_tools_widget)
        self.units_le = QLineEdit("mm")
        self.known_dist_le = QLineEdit("1.0") # Note: This is now just a reference for the dialog
        self.upp_le = QLineEdit("0.0")
        self.upp_le.setReadOnly(True)
        self.btn_calibrate = QPushButton("Calibrate...") # CHANGED
        calib_tools_layout.addRow("Units (e.g., mm):", self.units_le)
        calib_tools_layout.addRow("Known Distance:", self.known_dist_le)
        calib_tools_layout.addRow("Units Per Pixel:", self.upp_le)
        calib_tools_layout.addRow(self.btn_calibrate)
        calib_layout.addWidget(self.preview_label, 2)
        calib_layout.addWidget(calib_tools_widget, 1)
        step2_layout.addLayout(nav_layout)
        step2_layout.addWidget(calib_widget)
        self.step2_group.setEnabled(False)
        layout.addWidget(self.step2_group)

        # === STEP 3: PRESET ===
        self.step3_group = QGroupBox("Step 3: Choose Analysis Preset")
        preset_layout = QHBoxLayout(self.step3_group)
        self.preset_button_group = QButtonGroup(self)
        presets = ["Fast", "Balanced", "Accurate", "Robust (Low Texture)", "High Strain Gradient"]
        for preset_name in presets:
            rb = QRadioButton(preset_name)
            self.preset_button_group.addButton(rb)
            preset_layout.addWidget(rb)
            if preset_name == "Balanced": rb.setChecked(True)
        self.step3_group.setEnabled(False)
        layout.addWidget(self.step3_group)
        
        scroll_area.setWidget(content_widget)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.on_accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(self.button_box)
        
        # --- Connections ---
        btn_browse_video.clicked.connect(self._load_calibration_frames)
        btn_browse_files.clicked.connect(self._browse_files_and_load_frames)
        btn_browse_output.clicked.connect(self.browse_output_dir)
        self.btn_prev_frame.clicked.connect(self.show_prev_frame)
        self.btn_next_frame.clicked.connect(self.show_next_frame)
        self.btn_calibrate.clicked.connect(self._open_calibration_dialog) # CHANGED
        self.rb_interval.toggled.connect(self._update_extractor_options)
        self.rb_specific.toggled.connect(self._update_extractor_options)
        self.rb_range.toggled.connect(self._update_extractor_options)
        self.rb_video.toggled.connect(self._update_input_source_ui)
        
        self._update_extractor_options()
        self._update_input_source_ui()

    def _open_calibration_dialog(self): # NEW METHOD
        if not self.calibration_frame_paths:
            QMessageBox.warning(self, "No Image", "Please select an input source to load a calibration frame first.")
            return
        try:
            current_frame_path = self.calibration_frame_paths[self.current_frame_index]
            ref_img_np = iio.imread(current_frame_path)
            if ref_img_np.ndim == 3:
                ref_img_np = ref_img_np[..., 0]

            # Use the existing CalibrateUnitsDialog
            dlg = CalibrateUnitsDialog(ref_img_np, self)
            # Pass the current "known distance" value to the dialog
            dlg.ed_length.setValue(safe_float(self.known_dist_le.text(), 1.0))

            if dlg.exec():
                self.units_le.setText(dlg.units)
                self.upp_le.setText(f"{dlg.units_per_pixel:.8f}")
                QMessageBox.information(self, "Success", "Calibration complete.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load image for calibration.\n{e}")

    def show_current_frame(self): # REPAIRED
        if not self.calibration_frame_paths: return
        frame_path = self.calibration_frame_paths[self.current_frame_index]
        
        # FitLabel handles scaling. No need for drawing logic here anymore.
        self.preview_label.setPixmap(QPixmap(frame_path))
        
        self.lbl_frame_info.setText(f"Frame for Calibration: {self.current_frame_index + 1}/{len(self.calibration_frame_paths)}")
        self.btn_prev_frame.setEnabled(self.current_frame_index > 0)
        self.btn_next_frame.setEnabled(self.current_frame_index < len(self.calibration_frame_paths) - 1)

    # --- Other methods are mostly unchanged or removed ---
    def _update_input_source_ui(self):
        is_video = self.rb_video.isChecked()
        self.source_stack.setCurrentIndex(0 if is_video else 1)
        self.extraction_group.setEnabled(is_video)

    def _browse_files_and_load_frames(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Image Files", "", "Image Files (*.png *.jpg *.jpeg *.tif)")
        if paths:
            self.image_file_paths = sorted(paths, key=natural_key)
            self.files_path_label.setText(f"{len(paths)} file(s) selected.")
            self._load_calibration_frames()

    def _load_calibration_frames(self):
        self.calibration_frame_paths.clear()
        is_video = self.rb_video.isChecked()
        if is_video:
            path = self.video_path_le.text()
            if not path:
                path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
                if not path: return
                self.video_path_le.setText(path)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.lbl_frame_info.setText("Error: Could not open video."); return
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_positions = [int(total_frames * 0.2), int(total_frames * 0.5), int(total_frames * 0.8)]
            temp_dir = os.path.join(self.output_dir_le.text() or ".", "temp_calib_frames")
            os.makedirs(temp_dir, exist_ok=True)
            for i, frame_num in enumerate(frame_positions):
                if frame_num < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_path = os.path.join(temp_dir, f"calib_frame_{i}.png")
                        Image.fromarray(frame_rgb).save(frame_path)
                        self.calibration_frame_paths.append(frame_path)
            cap.release()
        else:
            all_files = self.image_file_paths
            if len(all_files) < 3: self.calibration_frame_paths = all_files
            else: self.calibration_frame_paths = [all_files[0], all_files[len(all_files) // 2], all_files[-1]]
        if self.calibration_frame_paths:
            self.step2_group.setEnabled(True); self.step3_group.setEnabled(True)
            self.current_frame_index = 0
            self.show_current_frame()
        else:
            self.lbl_frame_info.setText("No valid frames found.")
            self.step2_group.setEnabled(False); self.step3_group.setEnabled(False)

    def browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path: self.output_dir_le.setText(path)

    def _update_extractor_options(self):
        while self.extractor_options_layout.count():
            item = self.extractor_options_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        if self.rb_interval.isChecked():
            self.interval_le = QLineEdit("1.0")
            self.extractor_options_layout.addRow("Interval (seconds):", self.interval_le)
        elif self.rb_specific.isChecked():
            self.frames_le = QLineEdit("0, 100, 200")
            self.extractor_options_layout.addRow("Frame numbers:", self.frames_le)
        elif self.rb_range.isChecked():
            self.start_frame_le = QLineEdit("0"); self.end_frame_le = QLineEdit("100"); self.step_frame_le = QLineEdit("10")
            self.extractor_options_layout.addRow("Start frame:", self.start_frame_le)
            self.extractor_options_layout.addRow("End frame:", self.end_frame_le)
            self.extractor_options_layout.addRow("Step:", self.step_frame_le)

    def show_prev_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1; self.show_current_frame()

    def show_next_frame(self):
        if self.current_frame_index < len(self.calibration_frame_paths) - 1:
            self.current_frame_index += 1; self.show_current_frame()

    def on_accept(self):
        preset_name = self.preset_button_group.checkedButton().text()
        self._config_generator.generate_presets_from_image(self.calibration_frame_paths[self.current_frame_index])
        preset_config = self._config_generator.presets.get(preset_name)
        selected_mode = self.analysis_mode_combo.currentText()
        dic_analysis_config = "NO_UPDATE" if "Fixed Reference" in selected_mode else "KEEP_MOST_POINTS"
        if preset_config: preset_config['dic']['dic_config'] = dic_analysis_config
        input_settings = {}
        if self.rb_video.isChecked():
            input_settings["input_type"] = "video"; input_settings["video_path"] = self.video_path_le.text()
            if self.rb_interval.isChecked():
                input_settings["extraction_method"] = "Interval"; input_settings["extraction_interval"] = safe_float(self.interval_le.text(), 1.0)
            elif self.rb_specific.isChecked():
                input_settings["extraction_method"] = "Specific"; input_settings["extraction_frames"] = self.frames_le.text()
            elif self.rb_range.isChecked():
                input_settings["extraction_method"] = "Range"; input_settings["extraction_start_frame"] = safe_int(self.start_frame_le.text())
                input_settings["extraction_end_frame"] = safe_int(self.end_frame_le.text()); input_settings["extraction_step_frame"] = safe_int(self.step_frame_le.text(), 1)
        else:
            input_settings["input_type"] = "files"; input_settings["image_file_paths"] = self.image_file_paths
        self.config_data = {
            "output_dir": self.output_dir_le.text(),
            "ref_image_path_for_calib": self.calibration_frame_paths[self.current_frame_index],
            "dic_preset_name": preset_name, "dic_config": preset_config,
            "units": self.units_le.text(), "units_per_pixel": safe_float(self.upp_le.text(), 0.0)
        }
        self.config_data.update(input_settings)
        self.accept()

    def get_configuration(self):
        return self.config_data