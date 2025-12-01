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
import datetime
from pathlib import Path
from PySide6.QtWidgets import QApplication

def _read_text_safely(path: str) -> str:
    """
    Reads the QSS file with UTF-8, falling back to Latin-1 if decoding fails.
    This handles potential encoding issues, especially on Windows.
    """
    p = Path(path)
    try:
        # First, try to read with UTF-8 (handles BOMs correctly)
        return p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Fallback to Latin-1, which preserves all bytes without error.
        return p.read_text(encoding="latin-1")
    except FileNotFoundError:
        # Return an empty string if the file doesn't exist.
        return ""

class ThemeManager:
    def __init__(self):
        self.app = QApplication.instance()
        self.current_theme = 'light'
        self.light_theme_path = 'light_theme.qss'
        self.dark_theme_path = 'dark_theme.qss'

        light_base_qss = _read_text_safely(self.light_theme_path)
        dark_base_qss = _read_text_safely(self.dark_theme_path)

        self.light_qss = light_base_qss
        self.dark_qss  = dark_base_qss

        if not light_base_qss and not dark_base_qss:
            print("Warning: Theme files (light_theme.qss, dark_theme.qss) not found. Using default styles.")

    def get_system_theme(self):
        """Returns 'light' for daytime (6 AM to 6 PM), otherwise 'dark'."""
        now_hour = datetime.datetime.now().hour
        if 6 <= now_hour < 18:
            return 'light'
        else:
            return 'dark'

    def apply_theme(self, theme_name):
        """
        Applies a 'light' or 'dark' theme to the entire application.
        If the stylesheet is invalid, it will be cleared to prevent a crash.
        """
        stylesheet_to_apply = ""
        if theme_name == 'dark' and self.dark_qss:
            stylesheet_to_apply = self.dark_qss
            self.current_theme = 'dark'
        elif self.light_qss:
            stylesheet_to_apply = self.light_qss
            self.current_theme = 'light'

        try:
            self.app.setStyleSheet(stylesheet_to_apply)
        except Exception as e:
            print(f"Error applying stylesheet for '{theme_name}' theme: {e}")
            # An empty stylesheet is better than a broken one.
            self.app.setStyleSheet("")

        return self.current_theme

    def toggle_theme(self):
        """Switches from the current theme to the other."""
        return self.apply_theme('dark' if self.current_theme == 'light' else 'light')