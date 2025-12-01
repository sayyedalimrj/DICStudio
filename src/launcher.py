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

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QSpacerItem, QSizePolicy, QMessageBox, QFrame)
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtCore import QSize, Qt, QSettings

from manual_mode_window import ManualModeWindow
from automatic_config_dialog import AutomaticConfigDialog
from automation_controller import AutomationController
from plot_window import ViewPlotsWindow
from main_app import AnalysisProgressDialog

class LauncherWindow(QMainWindow):
    def __init__(self, theme_manager):
        super().__init__()
        self.theme_manager = theme_manager
        self.setWindowTitle("DIC Studio - Mirjafari")
        self.setWindowIcon(QIcon("logo.png"))
        self.setMinimumSize(500, 550) # Adjusted for new design
        g = self.screen().availableGeometry()
        self.resize(int(g.width() * 0.25), int(g.height() * 0.5))
        self.move(g.center() - self.rect().center())
        self.settings = QSettings()
        if (b := self.settings.value('launcher/geom')):
            self.restoreGeometry(b)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Header ---
        header_frame = QFrame()
        header_frame.setObjectName("header_frame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 10, 20, 10)

        # Logo Placeholder
        self.logo_label = QLabel()
        self.logo_label.setFixedHeight(48)
        self.logo_label.setObjectName("logo_label")
        # To add your logo, place a file named 'logo.png' in the same directory as run_app.py
        # and uncomment the next line:
        self.logo_label.setPixmap(QPixmap("logo.png").scaledToHeight(48, Qt.SmoothTransformation))

        title_layout = QVBoxLayout()
        title = QLabel("DIC Studio")
        title.setObjectName("header_title")
        subtitle = QLabel("Digital Image Correlation Software")
        subtitle.setObjectName("subtitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        self.theme_toggle_button = QPushButton()
        self.theme_toggle_button.setObjectName("theme_toggle_button")
        self.theme_toggle_button.setCheckable(True)
        self.theme_toggle_button.setFixedSize(32, 32)
        self.theme_toggle_button.clicked.connect(self.toggle_theme)

        header_layout.addWidget(self.logo_label)
        header_layout.addSpacing(15)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        header_layout.addWidget(self.theme_toggle_button)
        main_layout.addWidget(header_frame)

        # --- Content Area ---
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(40, 30, 40, 30)
        content_layout.setSpacing(25)

        # Manual Mode Card
        manual_card = self._create_mode_card(
            "Manual DIC Workflow",
            "Step-by-step control over preprocessing, DIC analysis, and post-processing.",
            "icon_manual.png", # You can create an icon for this
            self.open_manual_mode
        )

        # Automatic Mode Card
        automatic_card = self._create_mode_card(
            "Automatic DIC Workflow",
            "A streamlined, wizard-style process for quick and automated analysis.",
            "icon_auto.png", # And an icon for this
            self.start_automatic_mode
        )

        content_layout.addWidget(manual_card)
        content_layout.addWidget(automatic_card)
        main_layout.addLayout(content_layout)
        main_layout.addStretch()

        # --- Footer ---
        footer_frame = QFrame()
        footer_frame.setObjectName("footer_frame")
        footer_layout = QVBoxLayout(footer_frame)
        footer_layout.setContentsMargins(20, 10, 20, 10)
        author_label = QLabel("Developed by: Ali Mirjafari")
        email_label = QLabel("ali_mirjafari@civileng.iust.ac.ir")
        author_label.setAlignment(Qt.AlignCenter)
        email_label.setAlignment(Qt.AlignCenter)
        author_label.setObjectName("footer_label")
        email_label.setObjectName("footer_label")
        footer_layout.addWidget(author_label)
        footer_layout.addWidget(email_label)
        main_layout.addWidget(footer_frame)

        # --- Instance variables ---
        self.manual_window = None
        self.automation_controller = None
        self.plot_window = None
        self.update_theme_button()

    def _create_mode_card(self, title_text, description_text, icon_path, on_click):
        card = QFrame()
        card.setObjectName("mode_card")
        card_layout = QVBoxLayout(card)

        title_label = QLabel(title_text)
        title_label.setObjectName("card_title")

        description_label = QLabel(description_text)
        description_label.setWordWrap(True)
        description_label.setObjectName("card_description")

        button = QPushButton("Launch")
        button.clicked.connect(on_click)

        card_layout.addWidget(title_label)
        card_layout.addWidget(description_label)
        card_layout.addStretch()
        card_layout.addWidget(button, 0, Qt.AlignRight)

        # You can add an icon to the card if you like
        # icon_label = QLabel()
        # icon_label.setPixmap(QPixmap(icon_path).scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # main_hbox = QHBoxLayout()
        # main_hbox.addLayout(card_layout)
        # main_hbox.addWidget(icon_label)
        # card.setLayout(main_hbox)

        return card

    def update_theme_button(self):
        if self.theme_manager.current_theme == 'dark':
            self.theme_toggle_button.setIcon(QIcon('icon_light.png'))
            self.theme_toggle_button.setChecked(False)
        else:
            self.theme_toggle_button.setIcon(QIcon('icon_dark.png'))
            self.theme_toggle_button.setChecked(True)

    def toggle_theme(self):
        self.theme_manager.toggle_theme()
        self.update_theme_button()

    def open_manual_mode(self):
        self.manual_window = ManualModeWindow()
        self.manual_window.show()
        self.close()

    def handle_progress_update(self, percentage, message):
        if percentage >= 0:
            self.progress_dialog.set_indeterminate(False)
            self.progress_dialog.update_progress(percentage)
        else:
            self.progress_dialog.set_indeterminate(True)
        self.progress_dialog.update_message(message)

    def start_automatic_mode(self):
        config_dialog = AutomaticConfigDialog(self)
        if not config_dialog.exec():
            return

        config = config_dialog.get_configuration()
        is_video = config.get("input_type") == "video"
        is_files = config.get("input_type") == "files"
        path_provided = (is_video and config.get("video_path")) or \
                        (is_files and config.get("image_file_paths"))

        if not all([path_provided, config.get("output_dir"), config.get("ref_image_path_for_calib")]):
            QMessageBox.critical(self, "Error", "An Input Source, Output Directory, and a Calibration Image must be provided.")
            return

        self.manual_window = ManualModeWindow()
        self.progress_dialog = AnalysisProgressDialog(self)
        self.progress_dialog.setWindowTitle("Automatic Workflow in Progress")

        self.automation_controller = AutomationController(config)
        self.automation_controller.progress_update.connect(self.handle_progress_update)
        self.automation_controller.finished.connect(self.on_automation_finished)
        self.automation_controller.finished.connect(self.progress_dialog.accept)

        self.automation_controller.start()
        self.hide()
        self.progress_dialog.exec()

    def on_automation_finished(self, success, result_data):
        self.manual_window.show()
        self.manual_window.setEnabled(True)
        if success and "dic_results" in result_data:
            self.manual_window.processing_app.load_state_from_auto(result_data)
            QMessageBox.information(self, "Success", "Automatic workflow finished successfully!")
            self.plot_window = ViewPlotsWindow(
                parent=None,
                dic_out=result_data['dic_results'],
                strain_out=result_data['strain_results'],
                ref_image=result_data['ref_img'],
                image_paths=result_data['cur_paths'],
                units=result_data['dic_params'].get('units', ''),
                units_per_pixel=result_data['dic_params'].get('units_per_pixel', 0.0)
            )
            self.plot_window.show()
        elif success and "message" in result_data:
            QMessageBox.information(self, "Success", result_data["message"])
        else:
            error_message = result_data.get("error", "An unknown error occurred.")
            QMessageBox.critical(self, "Error", f"An error occurred during automation:\n{error_message}")

    def closeEvent(self, e):
        if hasattr(self, 'settings'):
            self.settings.setValue('launcher/geom', self.saveGeometry())
        super().closeEvent(e)