# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 05:28:18 2018

@author: Pranjal
"""

from xgboost import XGBRegressor
import catboost as cat
import lightgbm as lgb
from sklearn import metrics
from Data import get_data


x_train, x_test, y_train, y_test, submit_Id, submit_x = get_data()

XGB_params = {
    'objective': 'reg:linear',
    'silent': 0,
    'n_estimators' :500,
    'max_depth': 10,
    'learning_rate': 0.01,
    'min_child_weight':8,
    'gpu_id': 0,
    'max_bin': 16,
    'tree_method': 'gpu_hist'
}

LGBM_prams = {
    "max_depth": 15
}

Cat_prams = {
    'iterations':300,
    'l2_leaf_reg':1,
    'learning_rate':0.1,
    'depth': 13
}

xgb = XGBRegressor(**XGB_params)
lb = lgb.LGBMRegressor(**LGBM_prams)
cb = cat.CatBoostRegressor(**Cat_prams)

# XGBoost
print("Starting XGBoost...\n\n")
xgb.fit(x_train, y_train)

xgb_test_predicted = xgb.predict(x_test)
xgb_train_predicted = xgb.predict(x_train)
xgb_submit = xgb.predict(submit_x)

print("XGBoost Training Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_train, xgb_train_predicted))))
print("XGBoost Test Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_test, xgb_test_predicted))))
print("XGBoost Training Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_train, xgb_train_predicted)) * 100))
print("XGBoost Test Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_test, xgb_test_predicted)) * 100))

# LightGBM
print("Starting LightGBM...\n\n")
lb.fit(x_train, y_train)

lb_test_predicted = lb.predict(x_test)
lb_train_predicted = lb.predict(x_train)
lb_submit = lb.predict(submit_x)

print("LightGBM Training Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_train, lb_train_predicted))))
print("LightGBM Test Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_test, lb_test_predicted))))
print("LightGBM Training Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_train, lb_train_predicted)) * 100))
print("LightGBM Test Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_test, lb_test_predicted)) * 100))


# CatBoost
print("Starting CatBoost...\n\n")
cb.fit(x_train, y_train)

cb_test_predicted = cb.predict(x_test)
cb_train_predicted = cb.predict(x_train)
cb_submit = cb.predict(submit_x)

print("CatBoost Training Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_train, cb_train_predicted))))
print("CatBoost Test Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_test, cb_test_predicted))))
print("CatBoost Training Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_train, cb_train_predicted)) * 100))
print("CatBoost Test Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_test, cb_test_predicted)) * 100))


















