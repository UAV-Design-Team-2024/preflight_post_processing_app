import pandas as pd

from src.main_app import get_app

class DronePresetEditor:
    def __init__(self, motor_data):

        self.motor_dataframe = motor_data
        self.drone_df = None

    def build_df(self):
        app = get_app()
        self.drone_df = pd.DataFrame()

