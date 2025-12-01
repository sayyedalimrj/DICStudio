import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QGuiApplication

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'dependencies'))

from launcher import LauncherWindow
from theme_manager import ThemeManager

if __name__ == '__main__':
    QCoreApplication.setOrganizationName('Mirjafari')
    QCoreApplication.setApplicationName('DICStudio')
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)

    theme_manager = ThemeManager()
    theme_manager.apply_theme(theme_manager.get_system_theme())

    launcher = LauncherWindow(theme_manager)
    launcher.show()
    
    sys.exit(app.exec())