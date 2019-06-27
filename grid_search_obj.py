from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import file_helper

data = file_helper.open_pickled_file('data005.pickle')

_, _, X_train, X_test, y_train, y_test = data.prep_for_ann(split=.5)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# parameter_candidates = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
# ]

# parameter_candidates = [
#     {'n_estimators': [100, 500, 1000], 'bootstrap': [True], 'max_features': [2],
#      'max_depth': [4, 20, 100],
#      'min_samples_split': [1.0, 20], 'min_samples_leaf': [1, 10]},
# ]

parameter_candidates = [
    {'n_estimators': [100], 'bootstrap': [True], 'max_features': [2],
     'max_depth': [4],
     'min_samples_split': [1.0], 'min_samples_leaf': [1]},
]

from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
rf = RandomForestClassifier()
# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=rf, param_grid=parameter_candidates, cv=10, verbose=2)

# Train the classifier on data1's feature and target data
clf.fit(X_train, y_train)
# View the accuracy score
print('Best score for data1:', clf.best_score_)
# View the best parameters for the model found using grid search
print('Best n_estimators:', clf.best_estimator_.n_estimators)
print('Best bootstrap:', clf.best_estimator_.bootstrap)
print('Best max_features:', clf.best_estimator_.max_features)
print('Best max_depth:', clf.best_estimator_.max_depth)
print('Best min_samples_split:', clf.best_estimator_.min_samples_split)
print('Best min_samples_leaf:', clf.best_estimator_.min_samples_leaf)
