import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


class CrossVal:
    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

    def preprocess_data(self):
        self.data.dropna(inplace=True)
        self.x = self.data[self.cols_to_use]
        self.y = self.data['Price']

    def perform_cross_validation(self):
        my_pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor())
        scores = cross_val_score(my_pipeline, self.x, self.y, scoring='neg_mean_absolute_error')
        print('Mean Absolute Error %2f' % (-1 * scores.mean()))
        return scores


