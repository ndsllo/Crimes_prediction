import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import recall_score, precision_score


problem_title = 'Prediction of crimes'

Predictions = rw.prediction_types.make_regression(
    label_names=['prediction_crimes'])

workflow = rw.workflows.FeatureExtractorRegressor()

# define the score (specific score for the crimes problem)
class crime_error(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf
    
    def __init__(self, name='crime error',precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        max_true = np.log(np.maximum(1., y_true))
        max_pred = np.log(np.maximum(1., y_pred))
        loss = np.mean(np.abs(max_true - max_pred))
        
        return loss


class Precision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='prec', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        y_pred_binary = np.vectorize(lambda x : 0 if (x == 0) else 1)(y_pred)
        y_true_binary = np.vectorize(lambda x : 0 if (x == 0) else 1)(y_true)
        score = precision_score(y_true_binary, y_pred_binary)
        return score

class Recall(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='rec', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        y_pred_binary = np.vectorize(lambda x : 0 if (x == 0) else 1)(y_pred)
        y_true_binary = np.vectorize(lambda x : 0 if (x == 0) else 1)(y_true)
        score = recall_score(y_true_binary, y_pred_binary)
        return score

score_types = [
    crime_error(name='crime error', precision=2),
    # Precision and recall
    Precision(name='prec', precision=2),
    Recall(name='rec', precision=2),
]


# data source
path = {'data_train' : 'https://raw.githubusercontent.com/balldatascientist/Crimes_prediction/master/data/train/data_train.csv',
        'target_train' : 'https://raw.githubusercontent.com/balldatascientist/Crimes_prediction/master/data/train/target_train.csv', 
        'data_test' : 'https://raw.githubusercontent.com/balldatascientist/Crimes_prediction/master/data/test/data_test.csv', 
        'target_test': 'https://raw.githubusercontent.com/balldatascientist/Crimes_prediction/master/data/test/target_test.csv'}


def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=8, test_size=0.20, random_state=0)
    return cv.split(X,y,groups=X['Zip Code'])

def _read_data(urldata, urltarget):
    data = pd.read_csv(urldata, parse_dates=['Dispatch Date / Time'])
    target = pd.read_csv(urltarget)
    return data, target

# data pour visualisation/descriptive
def get_train_data():
    urldata, urltarget = path['data_train'], path['target_train']
    return _read_data(urldata, urltarget)

def get_test_data():
    urldata, urltarget = path['data_test'], path['target_test']
    return _read_data(urldata, urltarget)
		
	
# data externes: evenements_sportifs_usa and joursferiesmontgomery
def get_external_data(): 
    urla = 'https://raw.githubusercontent.com/balldatascientist/Crimes_prediction/master/data/external_data/evenements_sportifs_usa.csv'
    urlb = 'https://raw.githubusercontent.com/balldatascientist/Crimes_prediction/master/data/external_data/joursferiesmontgomery.csv'
    even_sport = pd.read_csv(urla, sep=';')
    jourferie = pd.read_csv(urlb, sep=';')
    return even_sport, jourferie 
	
