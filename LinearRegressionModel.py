import pandas as pd
import numpy as np  # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor


class LinearRegressionModel:
    def __init__(self, data):

        self.data = pd.read_csv(data)
        pd.set_option('display.max_columns', None)

    def print_data(self):
        print(self.data)

    def data_info(self):
        print(self.data.info())

    def data_head(self, n=5):
        print(self.data.head(n))

    def drop_columns(self, columns):
        self.data.drop(columns, axis=1, inplace=True)

    def separate_features_and_target(self, target):
        self.X = self.data.drop(target, axis=1)
        self.Y = self.data[target]

    def scale_features(self):
        col_for_scaled = []
        all_col_nm = self.X.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()

        for col in all_col_nm:
            unique_values = list(self.X[col].unique())
            if max(unique_values) > 2:
                col_for_scaled.append(col)

        x_cols_for_scale = self.X[col_for_scaled]
        scaler = StandardScaler()
        self.X[col_for_scaled] = scaler.fit_transform(x_cols_for_scale)

    def drop_nan_columns(self):
        nan_cols = self.X.columns[self.X.isna().any()].tolist()
        if nan_cols:
            print("Dropping columns with NaNs:", nan_cols)
            self.X.drop(nan_cols, axis=1, inplace=True)

    def encode_features(self):
        if 'Type' in self.X.columns:
            encoding = LabelEncoder()
            self.X["Type"] = encoding.fit_transform(self.X["Type"])

        numerical_cols = self.X.select_dtypes(include=['int', 'float']).columns.tolist()
        categorical_cols = self.X.select_dtypes(include=['object']).columns.tolist()

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_categorical = encoder.fit_transform(self.X[categorical_cols])
        encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_cols))

        self.X = pd.concat([self.X[numerical_cols], encoded_df], axis=1)

    def split_data(self, test_size=0.2, random_state=1232224):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=test_size,
                                                                                random_state=random_state)
        print("Data split into training and testing sets")

    def train_model(self):
        self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.x_test)
        r2 = r2_score(self.y_test, y_pred)
        print("R-squared (R2) Score:", r2)

    def build_neural_network(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200):
        self.nn_model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter)

    def train_neural_network(self):
        self.nn_model.fit(self.x_train, self.y_train)

    def evaluate_neural_network(self):
        y_pred = self.nn_model.predict(self.x_test)
        r2 = r2_score(self.y_test, y_pred)
        print("R-squared (R2) Score (Neural Network):", r2)

