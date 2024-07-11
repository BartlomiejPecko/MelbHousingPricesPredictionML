import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


class DataExploration:
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
        existing_cols = [col for col in columns if col in self.data.columns]
        if existing_cols:
            self.data.drop(existing_cols, axis=1, inplace=True)
            print(f"Usunięto kolumny: {existing_cols}")
        else:
            print("Żadne z podanych kolumn nie istnieją w zestawie danych.")

    def separate_features_and_target(self, target):
        self.X = self.data.drop(target, axis=1)
        self.Y = self.data[target]
        print(f"Podzielono cechy (X) i zmienną docelową (Y) na podstawie '{target}'")

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
        nan_cols_before = self.X.columns[self.X.isna().any()].tolist()
        if nan_cols_before:
            print("Kolumny z wartościami NaN przed usunięciem:", nan_cols_before)
            self.X.drop(nan_cols_before, axis=1, inplace=True)
            print(f"Usunięto kolumny z wartościami NaN: {nan_cols_before}")
        else:
            print("Nie znaleziono kolumn z wartościami NaN")


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

    def execute_data_exploration(self):
        self.print_data()
        self.data_info()
        self.data_head()
        self.drop_columns(['Address', 'Method', 'Date', 'Postcode', 'SellerG', 'Suburb'])
        self.separate_features_and_target('Price')
        self.scale_features()
        self.drop_nan_columns()
        self.encode_features()
        self.split_data()
        self.histogram()
