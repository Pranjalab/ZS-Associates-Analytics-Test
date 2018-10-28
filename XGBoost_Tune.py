# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 05:28:18 2018

@author: Pranjal
"""

from xgboost import XGBRegressor
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from Data import get_data


x_train, x_test, y_train, y_test, submit_Id, submit_x = get_data()

params_grid = {
       "max_depth": [8, 10, 20],
       'min_child_weight':[3,7,8],
       'n_estimators' : [500, 200]
       
}
params_fixed = {
    'objective': 'reg:linear',
    'learning_rate': 0.01,
    'gpu_id': 0,
    'max_bin':16,
    'tree_method':'gpu_hist'
}

bst_grid = GridSearchCV(
                        estimator=XGBRegressor(**params_fixed),
                        param_grid=params_grid,
                        cv = 3,
                        scoring='accuracy',
                        verbose=10, n_jobs=-1
                        )

bst_grid.fit(x_train, y_train)

test_predicted = bst_grid.predict(x_test)
train_predicted = bst_grid.predict(x_train)

print("Training Accuracy (R^2 score):" + str((metrics.r2_score(y_train, train_predicted))*100))
print("Test Accuracy (R^2 score):" + str((metrics.r2_score(y_test, test_predicted))*100))
print("Training Accuracy (R^2 score):" + str((metrics.mean_squared_error(y_train, train_predicted))*100))
print("Test Accuracy (R^2 score):" + str((metrics.mean_squared_error(y_test, test_predicted))*100))




















