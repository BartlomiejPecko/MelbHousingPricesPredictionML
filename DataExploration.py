import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


class DataExploration:
    def __init__(self, data):
        self.data = pd.read_csv(data)
        pd.set_option('display.max_columns', None)

    def print_data(self):
        print(self.data)

    def data_info(self):
        print(self.data.info())

    def show_na_val(self):
        na_rows = self.data[self.data.isna().any(axis=1)]
        print(na_rows)

    def drop_na_val(self):
        self.data.dropna(inplace=True)

    def data_describe(self):
        print(self.data.describe())


