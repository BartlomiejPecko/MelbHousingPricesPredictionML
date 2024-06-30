import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np


class Preprocessing:
    def __init__(self, data):
        self.data = pd.read_csv(data)
        pd.set_option('display.max_columns', None)
        self.train_data = None

    def histogram(self):
        x = self.data.drop(['Price'], axis=1)
        y = self.data['Price']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        train_data = x_train.join(y_train)
        train_data.hist(figsize=(10, 10), bins=20)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(15, 8))
        sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")
        plt.title('Correlation Heatmap')
        plt.show()

    def preprocess_train_data_columns(self):
        if self.train_data is None:
            raise ValueError("train_data is not initialized. Run histogram() method first.")

        self.train_data['Rooms'] = np.log(self.train_data['Rooms']) + 1
        # zrobic preprocessing