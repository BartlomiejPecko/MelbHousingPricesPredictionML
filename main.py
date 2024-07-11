import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from CrossVal import CrossVal
from LinearRegressionModel import LinearRegressionModel
from DataExploration import DataExploration


def main():

    data_path = 'melb_data.csv'

    cross_val_instance = None
    lr_instance = None
    de_instance = None
    pp_instance = None
    nn_instance = None

    while True:
        print("Select an action:")
        print("1: Perform Data Exploration")
        print("2: Perform Cross-Validation")
        print("3: Perform Linear-regression")
        print("4: Neural network")

        choice = input("Enter your choice (1-4): ")
        if choice == '1':
            de_instance = DataExploration('melb_data.csv')
            de_instance.execute_data_exploration()
        elif choice == '2':
            cross_val_instance = CrossVal(data_path)
            cross_val_instance.preprocess_data()
            scores = cross_val_instance.perform_cross_validation()
            print("Cross-validation scores:", scores)
        elif choice == '3':
            lr_instance = LinearRegressionModel('melb_data.csv')
            lr_instance.drop_columns(['Address', 'Method', 'Date', 'Postcode', 'SellerG', 'Suburb'])
            lr_instance.separate_features_and_target('Price')
            lr_instance.scale_features()
            lr_instance.drop_nan_columns()
            lr_instance.encode_features()
            lr_instance.split_data()
            lr_instance.train_model()
            lr_instance.evaluate_model()
            lr_instance.visualize_predictions()

        elif choice == '4':
            lr_instance = LinearRegressionModel('melb_data.csv')
            lr_instance.drop_columns(['Address', 'Method', 'Date', 'Postcode', 'SellerG', 'Suburb'])
            lr_instance.separate_features_and_target('Price')
            lr_instance.scale_features()
            lr_instance.drop_nan_columns()
            lr_instance.encode_features()
            lr_instance.split_data()
            lr_instance.build_neural_network(hidden_layer_sizes=(64, 32), max_iter=500)
            lr_instance.train_neural_network()
            lr_instance.evaluate_neural_network()
            lr_instance.visualize_predictions_nn()
        else:
            break

if __name__ == "__main__":
    main()