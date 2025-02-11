import logging
import asyncio
import socket

import PySide6.QtCore
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow
from PySide6.QtWidgets import QPushButton, QBoxLayout, QGridLayout, QVBoxLayout, QLabel

from main_app import get_app
logger = logging.getLogger()
logger.setLevel(0)

class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test Agricultural App Name")

        self.create_widgets()
        self.create_layout()

    def create_layout(self):

        self.flight_opt_section_offset = 2
        self.data_analysis_section_offset = 4

        qg = QGridLayout()
        qg.addWidget(self.drone_presets_label, 0, 1, 1, 1)
        qg.addWidget(self.create_drone_preset_btn, 1, 0, 1, 1)
        qg.addWidget(self.edit_drone_preset_btn, 1, 1, 1, 1)
        qg.addWidget(self.drone_presets_whats_this_hover, 1, 2, 1, 1)

        qg.addWidget(self.flight_opt_label, 0+self.flight_opt_section_offset, 1, 1, 1)
        qg.addWidget(self.create_flight_plan_btn, 1+self.flight_opt_section_offset, 1, 1, 1)

        qg.addWidget(self.data_analysis_server_label, 0+self.data_analysis_section_offset, 1, 1, 1)
        qg.addWidget(self.create_instance_btn, 1+self.data_analysis_section_offset, 0, 1, 1)
        qg.addWidget(self.go_to_db_btn, 1+self.data_analysis_section_offset, 1, 1, 1)
        qg.addWidget(self.settings_btn, 1+self.data_analysis_section_offset, 2, 1, 1)
        qg.addWidget(self.status_label, 2+self.data_analysis_section_offset, 1, 1, 1)
        qg.addWidget(self.ip_label, 3+self.data_analysis_section_offset, 1, 1, 1)

        self.setLayout(qg)
    def create_widgets(self):
        app = get_app()

        # region Drone Presets Section
        self.drone_presets_label = QLabel(f"Create or Edit Drone Characteristics")
        self.drone_presets_label.setAlignment(PySide6.QtCore.Qt.AlignCenter)

        self.create_drone_preset_btn = QPushButton("Create Drone Preset")
        self.create_drone_preset_btn.clicked.connect(app.create_drone_preset_window)
        self.edit_drone_preset_btn = QPushButton("Edit Drone Presets")
        self.drone_presets_whats_this_hover = QPushButton("?")
        self.drone_presets_whats_this_hover.setToolTip("Create a preset of physical characteristics for a drone. These"
                                                       " characteristics will be used to provide a first guess for total"
                                                       " flight time.")
        self.drone_presets_whats_this_hover.setEnabled(False)
        # endregion

        # region Flight Optimization Section

        self.flight_opt_label = QLabel(f"Create a Flight Path for Export")
        self.flight_opt_label.setAlignment(PySide6.QtCore.Qt.AlignCenter)

        self.create_flight_plan_btn = QPushButton("Create a Flight Plan")

        # endregion

        # region Data Analysis Section

        self.data_analysis_server_label = QLabel(f"Post-Flight Data Analysis")
        self.data_analysis_server_label.setAlignment(PySide6.QtCore.Qt.AlignCenter)

        self.create_instance_btn = QPushButton("Start Server")
        self.create_instance_btn.clicked.connect(app.create_app_instance)

        self.go_to_db_btn = QPushButton("Go to Database")
        self.settings_btn = QPushButton("Settings")

        self.status_label = QLabel(f"Web server is currently: {app.flask_server_status}")
        self.status_label.setAlignment(PySide6.QtCore.Qt.AlignCenter)

        self.ip_label = QLabel(f"Want to access from another machine on the same network?\nType {socket.gethostbyname(socket.gethostname())}:5000 into the machine's web browser.")
        self.ip_label.setHidden(True)
        self.ip_label.setAlignment(PySide6.QtCore.Qt.AlignCenter)

        # endregion
