# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 04:24:08 2018

@author: Pranjal
"""

# Initial imports
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cross_validation import train_test_split


def get_data(Remove_na_per = 100):
    
    # Importing the data
    train = pd.read_csv("data/train.csv")
    submit = pd.read_csv("data/test.csv")
    
    y = train['bestSoldierPerc']
    x = train.drop(['soldierId', 'shipId', 'attackId', 'bestSoldierPerc'], axis=1)
    
    submit_Id = submit['soldierId']
    submit_x = submit.drop(['soldierId', 'shipId', 'attackId'], axis=1)
    
    # Removing columns which has more then 60% of NaN value
    Removed_feature = []
    if Remove_na_per < 100:
        Total_feature = x.axes[1]
        print("Removing Null Values...")
        TN = x.shape[0]
        for feature in Total_feature:
            val = pd.isnull(x[feature]).sum()
            avg = (val / TN) * 100
            if avg > Remove_na_per:
                Removed_feature.append(feature)
                x = x.drop(feature, axis=1)
                submit_x = submit_x.drop(feature, axis=1)
        print("Shape of new train and test: ", x.shape, submit_x.shape)
        print("Feature removed are: ", str(Removed_feature))
    
    # Imputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    x = imputer.fit_transform(x)
    submit_x = imputer.fit_transform(submit_x)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
    
    return  x_train, x_test, y_train, y_test, submit_Id, submit_x










