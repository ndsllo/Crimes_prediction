
from __future__ import print_function

import os
from shutil import copyfile

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

# full URLS:
# data_train:  'https://raw.githubusercontent.com/balldatascientist/Crimes_prediction/master/data/train/data_train.csv',
# target_train:  'https://raw.githubusercontent.com/balldatascientist/Crimes_prediction/master/data/train/target_train.csv', 
# data_test:  'https://raw.githubusercontent.com/balldatascientist/Crimes_prediction/master/data/test/data_test.csv', 
# target_test: 'https://raw.githubusercontent.com/balldatascientist/Crimes_prediction/master/data/test/target_test.csv'}


URLBASE = 'https://raw.githubusercontent.com/balldatascientist/Crimes_prediction/master/data/{}'
URLS = ['train/data_train.csv', 'train/target_train.csv', 'test/data_test.csv', 'test/target_test.csv']
DATA = ['data_train.csv', 'target_train.csv', 'test_train.csv', 'target_test.csv']

def main(output_dir='data'):
    filenames = DATA
    full_urls = [URLBASE.format(url) for url in URLS]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for url, filename in zip(full_urls, filenames):
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            continue

        print("Downloading from {} ...".format(url))
        urlretrieve(url, filename=output_file)
        print("=> File saved as {}".format(output_file))

if __name__ == '__main__':
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()


"""
Ce commentaire sert de documentation du travail. 
Il contient certaines transformations préalables.

_________________________________________________
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split

# url source
# url = "https://raw.githubusercontent.com/balldatascientist/Crimes_prediction/master/data/source/crime.csv"
# df = pd.read_csv(url)

# def transform(df=df):

  # variable Crime
  # df.insert(0, 'Crime', 1) 
  
  # data columns to datetime
  # df['Dispatch Date / Time'] = pd.to_datetime(df['Dispatch Date / Time'], format='%m/%d/%Y %H:%M:%S %p').dt.date

  # select variables
  # cols_X = ['Incident ID', 'Offence Code', 'CR Number', 'Dispatch Date / Time',
       'NIBRS Code', 'Victims', 'Crime Name1', 'Crime Name2', 'Crime Name3',
       'Police District Name', 'Block Address', 'City', 'State', 'Zip Code',
       'Agency', 'Place', 'Sector', 'Beat', 'PRA', 'Address Number',
       'Street Prefix', 'Street Name', 'Street Suffix', 'Street Type',
       'Start_Date_Time', 'End_Date_Time', 'Latitude', 'Longitude',
       'Police District Number', 'Location']
  # cols_y = ['Dispatch Date / Time', 'Zip Code', 'Crime']
  # Selectionner seulement l'Etat de Maryland
  # df = df[df['State']=='MD'] 
  # y = df[cols_y]
  # X = df[cols_X]
  # data_train, data_test, target_train, target_test = train_test_split(X, y, test_size=0.33, random_state=0)
  # return data_train, data_test, target_train, target_test

# data_train, data_test, target_train, target_test = transform(df=df)
# save data
# data_train.to_csv("data_train.csv", index=False)
# target_train.to_csv("target_train.csv", index=False)
# data_test.to_csv("data_test.csv", index=False)
# target_test.to_csv("target_test.csv", index=False)

"""
