from grid_search_obj import grid_search_test
from random_forest_test import test_rfc
from file_helper import compare_labels
from file_helper import display_segmented_audio
from audio_file_obj import AudioData

# test many versions of Random Forest Classifiers on different data sets
data_candidates = None                  # default in grid_search_obj.py is DATA_CANDIDATES
data_candidate_strings = None           # default in grid_search_obj.py is DATA_CANDIDATE_STRINGS
parameter_candidates = None             # default in grid_search_obj.py is PARAMETER_CANDIDATES
test_track = False,                     # default in grid_search_obj.py is False
data_filename = None                    # default in grid_search_obj.py is DEFAULT_DATA_FILENAME = 'data017.pickle'
best_models = grid_search_test(data_candidates, data_candidate_strings, parameter_candidates, test_track, data_filename)

# test one Random Forest Classifier
data_filename = None                    # default in random_forest_test.py is DATA_FILENAME = 'data017.pickle'
test_split = None                       # default in random_forest_test.py is DEFAULT_TEST_SPLIT = .4
feat_labels = None                      # default in random_forest_test.py is FEAT_LABELS = ...
data_set_name = None                    # default in random_forest_test.py is DEFAULT_NAME = 'RFC test data'
test_rfc(data_filename, test_split, feat_labels, data_set_name)

# save a confusion matrix with % similarity between label sets as a png file
label1 = 'track10_labels.txt'
label2 = 'track10_labels_Rachel.txt'
compare_labels('track10', [label1, label2])

# for testing purposes, displays the wave file visually with labels filled
display_segmented_audio('track2.wav', 'track2_labels.txt')

# create and save a new data file
my_clip_size = None                 # default in audio_file_obj.py is DEFAULT_CLIP_SIZE = 500
my_window_size = None               # default in audio_file_obj.py is DEFAULT_WINDOW_SIZE = 2500 ie 5 seconds total
my_window_size2 = None              # default in audio_file_obj.py is DEFAULT_WINDOW_SIZE2 = 10000 ie 20 seconds total
data_types = ['mean', 'sd', 'mean_surr', 'sd_surr', 'mean_surr2', 'sd_surr2', 'sd_full', 'mean_full', 'label',
              'mean_left', 'mean_right', 'sd_left', 'sd_right', 'mean_surr_left', 'sd_surr_left']
data = AudioData(data_types, clip_size=my_clip_size, window_size=my_window_size, window_size2=my_window_size2)
data.print()
data.pickle_data()
