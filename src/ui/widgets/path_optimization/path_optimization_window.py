import logging
import asyncio
import socket
import matplotlib.pyplot as mpl

from typing import TYPE_CHECKING
import PySide6.QtCore
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow
from PySide6.QtWidgets import QPushButton, QBoxLayout, QGridLayout, QVBoxLayout, QLabel, QLineEdit, QComboBox, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from src.app_manager import get_app

logger = logging.getLogger()
logger.setLevel(0)

class PathOptimizationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Create and Optimize Drone Paths")

        self.create_widgets()
        self.create_layout()

    def create_layout(self):
        self.window_layout = QGridLayout()

        self.window_layout.addWidget(self.load_btn, 0, 0, 1, 1)
        self.window_layout.addWidget(self.canvas, 1, 1, 1, 1)
        self.window_layout.addWidget(self.toolbar, 2, 1, 1, 1)


        self.setLayout(self.window_layout)

    def create_widgets(self):
        app = get_app()
        self.figure = mpl.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.blank = QLabel("")
        self.load_btn = QPushButton("Load KML File (.kml)")
        self.load_btn.clicked.connect(self.load_data)


    def load_data(self):

        file = QFileDialog()
        file.setFileMode(QFileDialog.AnyFile)
        file.setNameFilter("KML Files (*.kml)")

        if file.exec_():
            self.kml_file = file.selectedFiles()




