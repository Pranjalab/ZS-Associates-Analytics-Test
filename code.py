

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
from sklearn.linear_model import LinearRegression
import numpy as np


x_train, x_test, y_train, y_test, submit_Id, submit_x = get_data()

XGB_params = {
    'objective': 'reg:linear',
    'silent': 0,
    'n_estimators' :300,
    'max_depth': 10,
    'learning_rate': 0.01,
    'min_child_weight':8,
    'gpu_id': 0,
    'max_bin': 16,
    'tree_method': 'gpu_hist'
}

LGBM_prams = {
    "max_depth": 15,
    'silent':False,
    'learning_rate': 0.1,
    'n_estimators':200,
    'num_leaves':300
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

# # XGBoost
# print("Starting XGBoost...\n\n")
# xgb.fit(x_train, y_train)
#
# xgb_test_predicted = xgb.predict(x_test)
# xgb_train_predicted = xgb.predict(x_train)
# xgb_submit = xgb.predict(submit_x)
#

# LightGBM
print("Starting LightGBM...\n\n")
lb.fit(x_train, y_train)

lb_test_predicted = lb.predict(x_test)
lb_train_predicted = lb.predict(x_train)
lb_submit = lb.predict(submit_x)


# CatBoost
print("Starting CatBoost...\n\n")
cb.fit(x_train, y_train)

cb_test_predicted = cb.predict(x_test)
cb_train_predicted = cb.predict(x_train)
cb_submit = cb.predict(submit_x)

#
# print("XGBoost Training Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_train, xgb_train_predicted))))
# print("XGBoost Test Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_test, xgb_test_predicted))))
# print("XGBoost Training Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_train, xgb_train_predicted)) * 100))
# print("XGBoost Test Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_test, xgb_test_predicted)) * 100))

print('\n')

print("LightGBM Training Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_train, lb_train_predicted))))
print("LightGBM Test Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_test, lb_test_predicted))))
print("LightGBM Training Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_train, lb_train_predicted)) * 100))
print("LightGBM Test Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_test, lb_test_predicted)) * 100))

print('\n')

print("CatBoost Training Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_train, cb_train_predicted))))
print("CatBoost Test Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_test, cb_test_predicted))))
print("CatBoost Training Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_train, cb_train_predicted)) * 100))
print("CatBoost Test Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_test, cb_test_predicted)) * 100))


# LinearRegression
print("\n\nStarting LinearRegression...\n\n")
LR_x_train = np.column_stack((lb_train_predicted, cb_train_predicted))
LR_x_test = np.column_stack((lb_test_predicted, cb_test_predicted))
LR_x_submit = np.column_stack((lb_submit, cb_submit))

LR = LinearRegression()
LR.fit(LR_x_train, y_train)

LR_test_predicted = LR.predict(LR_x_test)
LR_train_predicted = LR.predict(LR_x_train)
LR_submit = LR.predict(LR_x_submit)

print("LinearRegression Training Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_train, LR_train_predicted))))
print("LinearRegression Test Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_test, LR_test_predicted))))
print("LinearRegression Training Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_train, cb_train_predicted)) * 100))
print("LinearRegression Test Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_test, LR_test_predicted)) * 100))


with open('data/submission.csv', '+w') as file:
    file.write('soldierId,bestSoldierPerc\n')
    for i in range(len(LR_submit)):
        if LR_submit[i] > 1:
            LR_submit[i] = 1
        if LR_submit[i] < 0:
            LR_submit[i] = 0
        file.write(str(submit_Id[i]) + ',' + str(LR_submit[i]) + '\n')

