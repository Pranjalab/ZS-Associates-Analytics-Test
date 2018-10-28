import catboost as cb
from sklearn import metrics
from Data import get_data
from sklearn.model_selection import GridSearchCV

x_train, x_test, y_train, y_test, submit_Id, submit_x = get_data()

cat = cb.CatBoostRegressor(iterations=300, l2_leaf_reg=1, learning_rate=0.1)

param_dist = {'depth': [13]
          }

grid_search = GridSearchCV(cat, n_jobs=-1, param_grid=param_dist, cv = 2,
                           scoring="neg_mean_absolute_error", verbose=5)

grid_search.fit(x_train, y_train)

grid_search.best_estimator_


test_predicted = grid_search.predict(x_test)
train_predicted = grid_search.predict(x_train)


print("Training Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_train, train_predicted))*100))
print("Test Accuracy (mean_absolute_error):" + str((metrics.mean_absolute_error(y_test, test_predicted))*100))
print("Training Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_train, train_predicted))))
print("Test Accuracy (explained_variance_score):" + str((metrics.explained_variance_score(y_test, test_predicted))))


