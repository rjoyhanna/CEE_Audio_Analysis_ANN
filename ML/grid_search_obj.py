from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from file_helper import open_pickled_file
from random_forest_test import track_test
from random_forest_test import test_rfc_results

# below are arrays of data types that we are curious about testing
all_labels = ['mean_full', 'sd_full', 'mean_surr', 'sd', 'mean', 'sd_surr', 'mean_surr2', 'sd_surr2', 'label',
              'mean_left', 'mean_right', 'sd_left', 'sd_right', 'mean_surr_left', 'sd_surr_left']
sd_only = ['sd_full', 'sd', 'sd_surr', 'sd_surr2', 'label', 'sd_left', 'sd_right', 'sd_surr_left']
mean_only = ['mean_full', 'mean_surr', 'mean', 'mean_surr2', 'label', 'mean_left', 'mean_right', 'mean_surr_left']
most_important = ['mean_surr', 'sd_surr', 'mean_surr2', 'sd_surr2', 'label', 'mean_surr_left', 'sd_surr_left']
my_focus = ['mean_full', 'mean_surr', 'sd', 'mean', 'mean_surr2', 'label', 'mean_left', 'mean_right', 'mean_surr_left']
simple = ['mean_full', 'sd', 'mean', 'label', 'mean_left', 'mean_right', 'mean_surr_left']
dart = ['mean_surr', 'sd', 'label']

# an array of all the arrays of data types to be tested (these are sets of values that we are interested in)
DATA_CANDIDATES = [all_labels, sd_only, mean_only, most_important, my_focus, simple, dart]
DATA_CANDIDATE_STRINGS = ['all_labels', 'sd_only', 'mean_only', 'most_important', 'my_focus', 'simple', 'dart']

# these are the parameters we are tweaking to find the best Random Forest Classifier
PARAMETER_CANDIDATES = [
    {'n_estimators': [10],          # number of trees in each Random Forest Classifier
     'bootstrap': [True],               # bootstrapping is sampling data with replacement from the original
                                        # training set in order to produce multiple separate training sets
     'criterion': ['gini'],         # uses Gini impurity vs. “entropy” makes the split based on the information gain
     'max_features': [10],              # the max number of features considered when finding best split
     'max_depth': [None],           # how deep the tree can be
     'min_samples_split': [2],          # minimum num of samples that must be present in order for a split to occur
     'min_samples_leaf': [1],       # minimum size of the end node of each decision tree
     'min_weight_fraction_leaf': [0],   # like min_samples_leaf, uses a fraction of the sum total number of observations
     'max_leaf_nodes': [None],      # grows tree in best-first fashion, relative reduction in impurity
     'min_impurity_decrease': [0.],     # A node will be split if this split induces a decrease of the impurity
                                        # greater than or equal to this value
     'oob_score': [False],          # grabs all observations used in the trees and finds out the maximum score for
                                    # each observation base on the trees which did not use that observation to train
     'random_state': [0],
     'warm_start': [False],             # fits a new forest each time vs. reuses the solution of the previous fit
     'class_weight': [None]}        # weights of each class
]
# parameter_candidates = [
#     {'n_estimators': [10], 'bootstrap': [False, True], 'criterion': ['gini', 'entropy'],
#      'max_features': ['auto', 10, 4], 'max_depth': [None, 50000, 30000, 10000],
#      'min_samples_split': [2, .1, .01], 'min_samples_leaf': [1, 10],
#      'min_weight_fraction_leaf': [0, .2, .5, .8], 'max_leaf_nodes': [None, 500, 1000, 10000],
#      'min_impurity_decrease': [0., .01, .1, .5], 'oob_score': [True, False], 'random_state': [0],
#      'warm_start': [True, False], 'class_weight': ['balanced', None]}
# ]

# this is the test/train data location
DEFAULT_DATA_FILENAME = 'data017.pickle'

# save the RFC model for later
rf = RandomForestClassifier()


def grid_search_test(data_candidates=None, data_candidate_strings=None, parameter_candidates=None, test_track=False,
                     data_filename=None):
    if data_candidates is None:
        data_candidates = DATA_CANDIDATES
    if data_candidate_strings is None:
        data_candidate_strings = DATA_CANDIDATE_STRINGS
    if parameter_candidates is None:
        parameter_candidates = PARAMETER_CANDIDATES
    if data_filename is None:
        data_filename = DEFAULT_DATA_FILENAME

    best_model_strings = []
    overall_result_strings = []
    track_result_strings = []
    best_model_objs = []

    # for each set of data interested in testing
    for i in range(0, len(data_candidates)):
        data_set = data_candidates[i]
        data_set_name = data_candidate_strings[i]

        print('\nOPENING PICKLED DATA\n')
        # open the data and select the columns of interest
        data = open_pickled_file(data_filename)
        data.remove_cols(data_set)

        # scale and split the data
        _, _, X_train, X_test, y_train, y_test = data.prep_for_ann(split=.4)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create a Grid Search object with the classifier and parameter candidates
        clf = GridSearchCV(estimator=rf, param_grid=parameter_candidates, cv=10, verbose=2)

        # Train the classifier on data's feature and target data
        clf.fit(X_train, y_train)

        # View the accuracy score and params of the best model
        best_model_string = 'Best score for data: {}\nBest parameters: {}'.format(clf.best_score_, clf.best_params_)
        best_model = clf.best_estimator_

        # use the best model to predict the test set
        overall_result_string = test_rfc_results(best_model, X_test, y_test, data_set_name)

        # save most successful model, its performance, and its params
        best_model_strings.append(best_model_string)
        overall_result_strings.append(overall_result_string)
        best_model_objs.append(best_model)
        print(data_candidate_strings[i])
        print(data_candidates[i])
        print(best_model_strings[i])
        print(overall_result_strings[i])

        # used to get a sample performance of the best model on one specific wav file
        if test_track:
            track_result_string = track_test(best_model, data_set, data_set_name, scaler)
            track_result_strings.append(track_result_string)
            print(track_result_strings[i])

    # give detail of the most successful model, its performance, and its params for each data set
    for i in range(0, len(best_model_strings)):
        print(data_candidate_strings[i])
        print(data_candidates[i])
        print(best_model_strings[i])
        print(overall_result_strings[i])
        if test_track:
            print(track_result_strings[i])

    # return an array of Random Forest Classifier Objects (the most successful one for each dataset)
    return best_model_objs
