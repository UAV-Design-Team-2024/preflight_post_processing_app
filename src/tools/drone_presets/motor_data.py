import pandas as pd

class Motor_Data:
    def __init__(self, motor_data_file):
        self.df = pd.read_excel(motor_data_file[0])

        self.throttle_datasets = []
        self.thrust_datasets = []
        self.voltage_datasets = []
        self.current_datasets = []

        self.get_datasets()
    def find_keyword_col(self, keyword):
        count = 0
        for column in self.df.columns.tolist():
            if keyword.lower() in column.lower():
                if keyword == "thrust":
                    dataset = [str(cell/1000) for cell in self.df[column]]
                else:
                    dataset = [str(cell) for cell in self.df[column]]
                return dataset
            else:
                if count < 10:
                    count += 1
                else:
                    break

    def get_datasets(self):
        self.throttle_datasets = self.find_keyword_col("throttle")
        self.thrust_datasets = self.find_keyword_col("thrust")
        self.voltage_datasets = self.find_keyword_col("voltage")
        self.current_datasets = self.find_keyword_col("current")

