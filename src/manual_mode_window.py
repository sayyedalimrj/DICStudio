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
from PySide6.QtWidgets import QMainWindow, QTabWidget
from PySide6.QtCore import QSettings
from PySide6.QtGui import QIcon

# Import the main UI classes from your existing scripts
from Preprocessing import MainWindow as PreprocessingMainWindow
from main_app import MainWindow as ProcessingMainWindow
from point_inspector import PointInspectorApp

class ManualModeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manual DIC Mode | Mirjafari")
        self.setWindowIcon(QIcon("logo.png"))
        g = self.screen().availableGeometry()
        self.setMinimumSize(820, 560)
        self.resize(int(g.width()*0.82), int(g.height()*0.82))
        self.move(g.center() - self.rect().center())
        self.settings = QSettings()
        if (b := self.settings.value('manual/geom')): self.restoreGeometry(b)


        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Each "MainWindow" from your original files is added as a tab.
        # Since they are QMainWindow, we can take their central widget
        # to embed them correctly.
        
        # Tab 1: Preprocessing
        self.preprocessing_app = PreprocessingMainWindow()
        self.tabs.addTab(self.preprocessing_app.centralWidget(), "1. Preprocessing")

        # Tab 2: Processing
        self.processing_app = ProcessingMainWindow()
        self.tabs.addTab(self.processing_app.centralWidget(), "2. Processing")

        # Tab 3: Post-processing
        self.postprocessing_app = PointInspectorApp()
        self.tabs.addTab(self.postprocessing_app.centralWidget(), "3. Post-processing")

    def closeEvent(self, e):
        if hasattr(self, 'settings'):
            self.settings.setValue('manual/geom', self.saveGeometry())
        super().closeEvent(e)