import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from file_helper import compare_two_labels
from file_helper import create_label_track
from file_helper import open_pickled_file
from audio_file_obj import AudioData


FEAT_LABELS = ['mean', 'sd', 'mean_surr', 'mean_surr2', 'sd_full', 'mean_full',
               'mean_left', 'mean_right', 'sd_left', 'sd_right', 'mean_surr_left', 'label']
DATA_FILENAME = 'data017.pickle'
# this is the file we would use to get a sample performance of the RFC
DEFAULT_TEST_TRACK = 'track10'
DEFAULT_TEST_SPLIT = .4
DEFAULT_NAME = 'RFC test data'


# this is used to determine how many of each label the RFC guesses
def count_guesses(guesses):
    num_0 = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    for guess in guesses:
        if guess == 0:
            num_0 += 1
        elif guess == 1:
            num_1 += 1
        elif guess == 2:
            num_2 += 1
        elif guess == 3:
            num_3 += 1
    return num_0, num_1, num_2, num_3


def track_test(model, data_set, data_set_name, scaler, track_name=None):
    if track_name is None:
        track_name = DEFAULT_TEST_TRACK

    # create audio data from test track
    track_data = AudioData(data_set, files=[track_name], window_size=5000, window_size2=10000)

    # scale and split the data
    X_vals = scaler.transform(track_data.data.loc[:, track_data.data.columns != 'label'])
    y_test = track_data.data[['label']]

    # use the model to predict the label(s) of the data
    y_pred = model.predict(X_vals)
    y_pred = np.argmax(y_pred, axis=1, out=None)

    # determine how many of each label the RFC guesses
    num0, num1, num2, num3 = count_guesses(y_pred)

    # track the accuracy of the model on the given wav file
    track_result_string = '\nAccuracy: {:.2f}%\n\nlabeled 0: {}\nlabeled 1: {}\nlabeled 2: {}\nlabeled 3: {}' \
                          '\n'.format(accuracy_score(y_test, y_pred) * 100, num0, num1, num2, num3)

    # create a txt file that can be used as labels in Audacity
    create_label_track('{}_predicted_labels_{}.txt'.format(track_name, data_set_name), y_pred)

    return track_result_string


def test_rfc_results(model, X_test, y_test, data_set_name):
    # use the best model to predict the test set
    y_pred = model.predict(X_test)

    # convert the one hot encoded label data to single int label
    # ie [0 0 1 0] vs 2
    y_pred = np.argmax(y_pred, axis=1, out=None)
    y_test = np.argmax(y_test, axis=1, out=None)

    # determine how many of each label the RFC guesses
    num0, num1, num2, num3 = count_guesses(y_pred)

    # track the accuracy of the model on the given test set
    overall_result_string = '\nAccuracy: {:.2f}%\n\nlabeled 0: {}\nlabeled 1: {}\nlabeled 2: {}\nlabeled 3: ' \
                            '{}\n'.format(accuracy_score(y_test, y_pred) * 100, num0, num1, num2, num3)

    # save confusion matrix with accuracy percentage saved as a png
    compare_two_labels(y_test, y_pred, 'Actual', 'Predicted data017 {}'.format(data_set_name))
    return overall_result_string


def test_rfc(data_filename=None, test_split=None, feat_labels=None, data_set_name=None):
    if data_filename is None:
        data_filename = DATA_FILENAME
    if test_split is None:
        test_split = DEFAULT_TEST_SPLIT
    if feat_labels is None:
        feat_labels = FEAT_LABELS
    if data_set_name is None:
        data_set_name = DEFAULT_NAME
    if feat_labels[-1:] != 'label':
        raise ValueError('last data type must be \'label\'')
    X_labels = feat_labels[:-1]

    print('\nOPENING PICKLED DATA\n')
    # open the data and select the columns of interest
    data = open_pickled_file(data_filename)
    data.remove_cols(feat_labels)

    # scale and split the data
    _, _, X_train, X_test, y_train, y_test = data.prep_for_ann(split=test_split)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a Random Forest Classifier object
# FIX MEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    rfc = RandomForestClassifier(n_estimators=100, random_state=0, bootstrap=True)
    rfc.fit(X_train, y_train)

    # print the importance of each feature (type of data) used to make the trees
    print('Feature Importance:')
    for feature in zip(X_labels, rfc.feature_importances_):
        print(feature)

    # use the model to predict the test set
    overall_result_string = test_rfc_results(rfc, X_test, y_test, data_set_name)

    # used to get a sample performance of the best model on one specific wav file
    track_result_string = track_test(rfc, feat_labels, data_set_name, scaler)

    print(overall_result_string)
    print(track_result_string)

    return rfc
