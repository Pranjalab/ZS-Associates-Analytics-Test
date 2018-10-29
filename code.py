

# Initial imports
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def get_data(Remove_na_per = 100):
    
    # Importing the data
    train = pd.read_csv("data/train.csv")
    submit = pd.read_csv("data/test.csv")

    train = train[np.isfinite(train['bestSoldierPerc'])]
    
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



from xgboost import XGBRegressor
import catboost as cat
import lightgbm as lgb
from sklearn import metrics
from Data import get_data
import numpy as np


x_train, x_test, y_train, y_test, submit_Id, submit_x = get_data()


# LightGBM
print("Starting LightGBM...\n\n")

seed = 0


LGBM_prams = {
    "max_depth": 15,
    'max_bin': 255,
    'silent': False,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'num_leaves': 300,
    "device": "gpu"
}

lb = lgb.LGBMRegressor(**LGBM_prams, seed=seed)
lb.fit(x_train, y_train, verbose=True)

lb_test_predicted = lb.predict(x_test)
lb_train_predicted = lb.predict(x_train)
lb_submit = lb.predict(submit_x)

print('\n')

print("LightGBM Training Accuracy (mean_absolute_error):" + str(
    (metrics.mean_absolute_error(y_train, lb_train_predicted))))
print("LightGBM Test Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_test, lb_test_predicted))))
print("LightGBM Training Accuracy (explained_variance_score):" + str(
    (metrics.explained_variance_score(y_train, lb_train_predicted)) * 100))
print("LightGBM Test Accuracy (explained_variance_score):" + str(
    (metrics.explained_variance_score(y_test, lb_test_predicted)) * 100))


# CatBoost
print("Starting CatBoost...\n\n")


Cat_prams = {
    'iterations': 800,
    'l2_leaf_reg': 10,
    'learning_rate': 0.1,
    'depth': 13,
    'task_type':"GPU"
}


cb = cat.CatBoostRegressor(**Cat_prams, random_seed=seed)
cb.fit(x_train, y_train, verbose = False)

cb_test_predicted = cb.predict(x_test)
cb_train_predicted = cb.predict(x_train)
cb_submit = cb.predict(submit_x)

print('\n')

print("CatBoost Training Accuracy (mean_absolute_error):" + str(
    (metrics.mean_absolute_error(y_train, cb_train_predicted))))
print("CatBoost Test Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_test, cb_test_predicted))))
print("CatBoost Training Accuracy (explained_variance_score):" + str(
    (metrics.explained_variance_score(y_train, cb_train_predicted)) * 100))
print("CatBoost Test Accuracy (explained_variance_score):" + str(
    (metrics.explained_variance_score(y_test, cb_test_predicted)) * 100))


print("\n\nStarting XGBoost...\n\n")

predicted = [lb_train_predicted, cb_train_predicted, lb_test_predicted, cb_test_predicted, lb_submit, cb_submit]
for prad in predicted:
    for i in range(len(prad)):
        if prad[i] > 1:
            prad[i] = 1
        if prad[i] < 0:
            prad[i] = 0


XGB_x_train = np.column_stack((lb_train_predicted, cb_train_predicted))
XGB_x_test = np.column_stack((lb_test_predicted, cb_test_predicted))
XGB_x_submit = np.column_stack((lb_submit, cb_submit))

XGB_params = {
    'objective': 'reg:linear',
    'silent': 0,
    'n_estimators': 50,
    'subsample': 0.9,
    'colsample_bytree ': 0.4,
    'gpu_id': 0,
    'tree_method': 'gpu_hist'
}

eval_set =  [(XGB_x_train, y_train), (XGB_x_test, y_test)]
eval_metric = ["map","mae"]

XGB = XGBRegressor(**XGB_params, seed=seed)
XGB.fit(XGB_x_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=False)

XGB_test_predicted = XGB.predict(XGB_x_test)
XGB_train_predicted = XGB.predict(XGB_x_train)
XGB_submit = XGB.predict(XGB_x_submit)

print("XGBoost Training Accuracy (mean_absolute_error):" + str(
    (metrics.mean_absolute_error(y_train, XGB_train_predicted))))
print("XGBoost Test Accuracy (mean_absolute_error):" + str(
    (metrics.mean_absolute_error(y_test, XGB_test_predicted))))
print("XGBoost Training Accuracy (explained_variance_score):" + str(
    (metrics.explained_variance_score(y_train, cb_train_predicted)) * 100))
print("XGBoost Test Accuracy (explained_variance_score):" + str(
    (metrics.explained_variance_score(y_test, XGB_test_predicted)) * 100))
print("XGBoost Training Accuracy (r2_score):" + str(
    (metrics.r2_score(y_train, cb_train_predicted)) * 100))
print("XGBoost Test Accuracy (r2_score):" + str(
    (metrics.r2_score(y_test, XGB_test_predicted)) * 100))

acc = metrics.r2_score(y_test, XGB_test_predicted)
with open('data/submission_' + str(acc) + '.csv', '+w') as file:
    file.write('soldierId,bestSoldierPerc\n')
    for i in range(len(XGB_submit)):
        if XGB_submit[i] > 1:
            XGB_submit[i] = 1
        if XGB_submit[i] < 0:
            XGB_submit[i] = 0
        file.write(str(submit_Id[i]) + ',' + str(XGB_submit[i]) + '\n')


xgb = XGBRegressor()
xgb.fit(x_train, y_train)

import xgboost
ax = xgboost.plot_importance(xgb)
fig = ax.figure
fig.set_size_inches(15, 15)