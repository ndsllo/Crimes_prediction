from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import datetime
import math
import pandas as pd

class FeatureExtractor(object):
    def __init__(self):
        pass
    
    def fit(self, X, y):        
        pass
    
    def grouping_by_week(self, X, y):
        df = X.copy()
        df['crime_nb'] = y
        df['Date'] = df['Dispatch Date / Time']
        df1 = df.groupby(['Zip Code',pd.Grouper(key='Date', freq='W-MON')])['crime_nb'].sum().reset_index().sort_values('Date', ascending=False).reset_index()
        X_train = df1[['Zip Code','Date']]
        y_train = df1['crime_nb']
        return X, y
    
    def extract_function(self, X):
        df = X.copy()
        df['Zip Code'] = pd.to_numeric(df['Zip Code'], errors='coerce')
        df['Zip Code'] = df['Zip Code'].fillna(df['Zip Code'].median())
        df['Date'] = pd.to_datetime(df['Date'])
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        df = df.drop(columns=['Date'])
        return df
    
    def transform(self,X):
        return self.extract_function(X).reset_index().values