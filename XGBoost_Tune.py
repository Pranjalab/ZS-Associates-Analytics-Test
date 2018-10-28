# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 05:28:18 2018

@author: Pranjal
"""

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from Data import get_data


x_train, x_test, y_train, y_test, submit_Id, submit_x = get_data()

params_grid = {
       "max_depth": [20, 15],
       'min_child_weight':[9, 8],

}

params_fixed = {
    'objective': 'reg:linear',
    'learning_rate': 0.01,
    'n_estimators' :200,
    'gpu_id': 0,
    'max_bin':16,
    'tree_method':'gpu_hist'
}

grid_search = GridSearchCV(
                        estimator=XGBRegressor(**params_fixed),
                        param_grid=params_grid,
                        cv = 2,
                        verbose=5, n_jobs=-1,
                        scoring='neg_mean_absolute_error'
                        )

grid_search.fit(x_train, y_train)


grid_search.best_estimator_

test_predicted = grid_search.predict(x_test)
train_predicted = grid_search.predict(x_train)


print("Training Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_train, train_predicted))*100))
print("Test Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_test, test_predicted))*100))
print("Training Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_train, train_predicted))))
print("Test Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_test, test_predicted))))





















