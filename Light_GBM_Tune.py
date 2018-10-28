# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 05:28:18 2018

@author: Pranjal
"""

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from Data import get_data


x_train, x_test, y_train, y_test, submit_Id, submit_x = get_data()

lg = lgb.LGBMRegressor(silent=False, learning_rate=0.1, n_estimators=200, num_leaves=300)

param_dist = {"max_depth": [15, 16, 18]
}

grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 2,
                           scoring="neg_mean_absolute_error", verbose=5)

grid_search.fit(x_train, y_train)

grid_search.best_estimator_


test_predicted = grid_search.predict(x_test)
train_predicted = grid_search.predict(x_train)


print("Training Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_train, train_predicted))*100))
print("Test Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_test, test_predicted))*100))
print("Training Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_train, train_predicted))))
print("Test Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_test, test_predicted))))

