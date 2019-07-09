from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import file_helper
from audio_file_obj import AudioFile
from audio_file_obj import AudioData
from sklearn.metrics import accuracy_score


feat_labels = ['mean_full', 'sd_full', 'mean_surr', 'sd', 'mean', 'sd_surr', 'mean_surr2', 'sd_surr2', 'label']
# feat_labels = ['mean_surr', 'sd', 'mean', 'sd_surr', 'mean_surr2', 'sd_surr2', 'label']
X_labels = ['mean_surr', 'sd', 'mean', 'sd_surr', 'mean_surr2', 'sd_surr2']

data = file_helper.open_pickled_file('data016.pickle')
# data.remove_cols(feat_labels)

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
    {'n_estimators': [100], 'bootstrap': [False, True], 'max_features': [8],
     'max_depth': [20],
     'min_samples_split': [1.0], 'min_samples_leaf': [1]},
]

from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
rf = RandomForestClassifier()
# Create a classifier object with the classifier and parameter candidates
# clf = GridSearchCV(estimator=rf, param_grid=parameter_candidates, cv=10, verbose=2)
#
# # Train the classifier on data1's feature and target data
# clf.fit(X_train, y_train)
# # View the accuracy score
# print('Best score for data:', clf.best_score_)
# # View the best parameters for the model found using grid search
# print('Best n_estimators:', clf.best_estimator_.n_estimators)
# print('Best bootstrap:', clf.best_estimator_.bootstrap)
# print('Best max_features:', clf.best_estimator_.max_features)
# print('Best max_depth:', clf.best_estimator_.max_depth)
# print('Best min_samples_split:', clf.best_estimator_.min_samples_split)
# print('Best min_samples_leaf:', clf.best_estimator_.min_samples_leaf)


# clf = RandomForestClassifier(n_estimators=100, random_state=0, bootstrap=True)
clf = RandomForestClassifier(n_estimators=100, random_state=0, bootstrap=True, class_weight="balanced")
clf.fit(X_train, y_train, )
# print('Best score for data:', clf.oob_score_)
print('Feature Importance:')
for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)
y_pred = clf.predict(X_test)
print('\nAccuracy: {:.2f}%\n'.format(accuracy_score(y_test, y_pred) * 100))
num0 = 0
num1 = 0
num2 = 0
num3 = 0
for val in y_pred:
    label = np.argmax(val, axis=0, out=None)
    # print(label)
    if label == 0:
        num0 += 1
    elif label == 1:
        num1 += 1
    elif label == 2:
        num2 += 1
    elif label == 3:
        num3 += 1

print('labeled 0: {}\nlabeled 1: {}\nlabeled 2: {}\nlabeled 3: {}\n'.format(num0, num1, num2, num3))

y_pred = np.argmax(y_pred, axis=1, out=None)
y_test = np.argmax(y_test, axis=1, out=None)
from file_helper import compare_two_labels
compare_two_labels(y_test, y_pred, 'Actual', 'Predicted data016 weighted')
# compare_two_labels(y_test, y_pred, 'Actual', 'Predicted data016')

track01 = AudioData(['mean_full', 'sd_full', 'mean_surr', 'sd', 'mean', 'sd_surr', 'mean_surr2', 'sd_surr2'],
                    files=['track01'], window_size=5000, window_size2=10000)
# track01.remove_cols(X_labels)

track01 = scaler.transform(track01.data)
y_pred = clf.predict(track01)
from file_helper import create_label_track
y_pred = np.argmax(y_pred, axis=1, out=None)

num0 = 0
num1 = 0
num2 = 0
num3 = 0
for val in y_pred:
    label = val
    # print(label)
    if label == 0:
        num0 += 1
    elif label == 1:
        num1 += 1
    elif label == 2:
        num2 += 1
    elif label == 3:
        num3 += 1

print('labeled 0: {}\nlabeled 1: {}\nlabeled 2: {}\nlabeled 3: {}\n'.format(num0, num1, num2, num3))
create_label_track('track01_predicted_labels_weighted.txt', y_pred)
# trees = clf.estimators_
# i = 0
# import sklearn
# from sklearn import tree
# for dtree in trees:
#     i += 1
    # print('TREE #{}:'.format(i))
    # print('Number of nodes: {}'.format(dtree.tree_.node_count))
    # print('Features: {}\n'.format(dtree.tree_.feature))
    # tree.plot_tree(dtree)
