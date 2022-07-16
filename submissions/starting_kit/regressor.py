from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator

import numpy as np

class Regressor(BaseEstimator):
    def __init__(self):
     self.rgr = regressor = RandomForestRegressor(n_estimators=15)

    def fit(self, X, y):
        self.rgr.fit(X, np.ravel(y))

    def predict(self, X):
        y_predicted = self.rgr.predict(X)
        #y_predicted should be 2D with 1 columns
        y_predicted = y_predicted.reshape((len(y_predicted), 1))
        return y_predicted