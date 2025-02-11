import logging
import asyncio
import socket
from typing import TYPE_CHECKING
import PySide6.QtCore
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow
from PySide6.QtWidgets import QPushButton, QBoxLayout, QGridLayout, QVBoxLayout, QLabel, QLineEdit, QComboBox, QFileDialog



from tools.drone_presets.motor_data import Motor_Data
from tools.drone_presets.drone_preset import DronePresetEditor
from src.app_manager import get_app

logger = logging.getLogger()
logger.setLevel(0)

class DronePresetWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Create/Edit Drone Presets")

        self.create_widgets()
        self.create_layout()

    def create_layout(self):
        self.window_layout = QGridLayout()
        self.window_layout.addWidget(self.load_btn, 0, 1, 1, 1)
        self.window_layout.addWidget(self.starting_label, 0, 1, 1, 1)

        self.window_layout.addWidget(self.section_1_label, 1, 1, 1, 1)
        self.window_layout.addWidget(self.throttle_label, 2, 1, 1, 1)
        self.window_layout.addWidget(self.voltage_label, 3, 1, 1, 1)
        self.window_layout.addWidget(self.current_label, 4, 1, 1, 1)
        self.window_layout.addWidget(self.thrust_label, 5, 1, 1, 1)
        self.window_layout.addWidget(self.optional_section_label,6, 1, 1, 1)
        self.window_layout.addWidget(self.battery_capacity_label, 7, 1, 1, 1)
        self.window_layout.addWidget(self.RPM_label, 8, 1, 1, 1)

        self.window_layout.addWidget(self.throttle_input, 2, 2, 1, 1)
        self.window_layout.addWidget(self.voltage_input, 3, 2, 1, 1)
        self.window_layout.addWidget(self.current_input, 4, 2, 1, 1)
        self.window_layout.addWidget(self.thrust_input, 5, 2, 1, 1)
        self.window_layout.addWidget(self.battery_capacity_input, 7, 2, 1, 1)
        self.window_layout.addWidget(self.RPM_input, 8, 2, 1, 1)

        self.window_layout.addWidget(self.section_2_label, 9, 1, 1, 1)
        self.window_layout.addWidget(self.drone_weight_label, 10, 1, 1, 1)
        self.window_layout.addWidget(self.tank_capacity_label, 11, 1, 1, 1)

        self.window_layout.addWidget(self.drone_weight_input, 10, 2, 1, 1)
        self.window_layout.addWidget(self.tank_capacity_input, 11, 2, 1, 1)

        self.window_layout.addWidget(self.blank, 12, 1, 1, 1)

        self.window_layout.addWidget(self.preset_name_label, 13, 1, 1, 1)
        self.window_layout.addWidget(self.preset_name_input, 13, 2, 1, 1)
        self.window_layout.addWidget(self.save_btn, 13, 3, 1, 1)

        self.setLayout(self.window_layout)

    def create_widgets(self):
        app = get_app()

        self.blank = QLabel("")
        self.load_btn = QPushButton("Load Motor Data (.xlsx)")
        self.load_btn.clicked.connect(self.load_data)

        self.starting_label = QLabel(f"Please verify that the below information matches manufacturer data:")
        self.starting_label.setStyleSheet("font-weight: bold")
        self.starting_label.setHidden(True)

        self.section_1_label = QLabel("Battery and Propulsion Specs")
        self.section_1_label.setStyleSheet("font-weight: bold")

        self.throttle_label = QLabel("Throttle (%)")
        self.thrust_label = QLabel(f"Thrust ({app.units.mass_unit}/force)")
        self.voltage_label = QLabel("Voltage (Volts)")
        self.current_label = QLabel("Current (Amps)")


        self.optional_section_label = QLabel(f"Please input EITHER: a rough estimate OR manufacturer provided data")
        self.optional_section_label.setStyleSheet("font-weight: bold")

        self.battery_capacity_label = QLabel("Singular Battery Capacity (Amp Hours (Ah))")
        self.RPM_label = QLabel("Singular Propeller RPM")

        self.throttle_input = QComboBox()
        self.thrust_input = QLineEdit()
        self.voltage_input = QLineEdit()
        self.current_input = QLineEdit()
        self.battery_capacity_input = QLineEdit()
        self.RPM_input = QLineEdit()

        self.throttle_input.setEnabled(False)
        self.throttle_input.currentIndexChanged.connect(self.update_section_1_layouts)

        self.thrust_input.setReadOnly(True)
        self.voltage_input.setReadOnly(True)
        self.current_input.setReadOnly(True)


        self.section_2_label = QLabel("Please enter payload specifications:")
        self.section_2_label.setStyleSheet("font-weight: bold")

        self.drone_weight_label = QLabel(f"Drone Weight ({app.units.mass_unit})")
        self.tank_capacity_label = QLabel(f"Tank Capacity ({app.units.mass_unit})")

        self.drone_weight_input = QLineEdit()
        self.tank_capacity_input = QLineEdit()

        self.preset_name_label = QLabel("Preset Name:")
        self.preset_name_input = QLineEdit()

        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)


    def save_data(self):
        app = get_app()
        self.preset_unit = app.units.unit_name
        self.preset_name = self.preset_name_input.text()

        self.preset_drone_weight = float(self.drone_weight_input.text())
        self.preset_tank_capacity = float(self.tank_capacity_input.text())
        self.preset_battery_capacity = float(self.battery_capacity_input.text())
        self.preset_propeller_rpm = float(self.RPM_input.text())



    def load_data(self):

        file = QFileDialog()
        file.setFileMode(QFileDialog.AnyFile)
        file.setNameFilter("Excel Files (*.xlsx)")

        if file.exec_():
            data_file = file.selectedFiles()

        self.motor_dataset = Motor_Data(data_file)

        self.window_layout.removeWidget(self.load_btn)
        self.window_layout.addWidget(self.load_btn, 0, 3, 1, 1)

        self.starting_label.setHidden(False)

        self.throttle_input.setEnabled(True)
        self.throttle_input.clear()
        self.throttle_input.addItems(self.motor_dataset.throttle_datasets)
        self.throttle_input.setCurrentIndex(0)

        self.setLayout(self.window_layout)

    def update_section_1_layouts(self):
        index = self.throttle_input.currentIndex()

        self.voltage_input.setText(f"{self.motor_dataset.voltage_datasets[index]}")
        self.thrust_input.setText(f"{self.motor_dataset.thrust_datasets[index]}")
        self.current_input.setText(f"{self.motor_dataset.current_datasets[index]}")
