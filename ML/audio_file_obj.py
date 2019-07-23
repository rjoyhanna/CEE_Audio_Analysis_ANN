from .file_helper import add_wav
from .file_helper import get_valid_audio_files
from .file_helper import get_last_data_num
from .file_helper import DATA_FOLDER_PATH
from .file_helper import add_txt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import librosa
import numpy as np
import pickle

SUPPORTED_DATA_TYPES = ['fourier', 'fourier_surr', 'mean', 'sd', 'mean_surr', 'sd_surr', 'mean_surr2', 'sd_surr2',
                        'sd_full', 'mean_full', 'label', 'mean_left', 'mean_right', 'sd_left', 'sd_right',
                        'mean_surr_left', 'sd_surr_left']
DEFAULT_CLIP_SIZE = 500  # half second total
DEFAULT_WINDOW_SIZE = 2500  # 5 seconds total
DEFAULT_WINDOW_SIZE2 = 10000  # 20 seconds total
MILL_TO_SEC = 1000
SEED = 7


# takes start and end times in milliseconds, and a data frame of handwritten labels as arguments
# returns the label that accompanies the given timestamp
# labels are as follows:
#   *  3 means other
#   *  0 means no voice
#   *  1 means one voice
#   *  2 means multiple voice
#   * -1 means transition (unusable)
def get_label(start, end, labels):
    end = end / MILL_TO_SEC
    start = start / MILL_TO_SEC
    last_label = labels.shape[0]
    for i in range(0, last_label):
        if labels.loc[i, 'end'] >= end:  # if the clip ends before the current label does
            if labels.loc[i, 'start'] <= start:  # if the clip starts after the current label does
                return labels.loc[i, 'label']
            else:  # fix this to account for incomplete labels
                if labels.loc[i - 1, 'label'] == 0:  # ignore silences in favor of sounds
                    return labels.loc[i, 'label']
                elif labels.loc[i, 'label'] == 0:  # ignore silences in favor of sounds
                    return labels.loc[i - 1, 'label']
                len_left = labels.loc[i-1, 'end'] - labels.loc[i-1, 'start']
                len_right = labels.loc[i, 'end'] - labels.loc[i, 'start']
                if len_left >= len_right:
                    return labels.loc[i-1, 'label']  # this is a transition label (the clip contains multiple labels)
                else:
                    return labels.loc[i, 'label']  # this is a transition label (the clip contains multiple labels)
    return -1  # remove rows with missing labels


# use this to ignore smaller labels and make the labels more general
def simplify_labels(labels, threshold=1):
    last_label = labels.shape[0]
    for i in range(0, last_label):
        if labels.loc[i, 'end'] - labels.loc[i, 'start'] < threshold:
            curr_label = labels.loc[i, 'label']
            if i > 0:
                if i < last_label - 1:
                    left_label = labels.loc[i - 1, 'label']
                    right_label = labels.loc[i + 1, 'label']
                    if left_label == right_label:
                        labels.loc[i, 'label'] = left_label
                    # elif left_label == 0 and curr_label == :
    return labels


# filename should not include an extension
# This is the object that holds data for ONE WAV FILE
class AudioFile:
    def __init__(self, filename):
        self.name = filename
        self.labels = pd.read_csv(add_txt(self.name), delimiter="\t", names=["start", "end", "label"])
        # self.labels = simplify_labels(pd.read_csv(add_txt(self.name), delimiter="\t", names=["start", "end", "label"]),
        #                               threshold=1)
        self.audio, self.sr = librosa.load(add_wav(filename))

    # returns the data based on the energy of the audio
    @staticmethod
    def get_rmse(clip, data_type):
        rms = librosa.feature.rmse(y=clip, frame_length=1, hop_length=1).flatten()
        if 'mean' == data_type:
            mean = np.mean(rms)
            return mean
        elif 'sd' == data_type:
            sd = np.std(rms)
            return sd

    # creates a dataframe of specified data types for ONE WAV FILE
    def build_dataframe(self, data_types, clip_size, window_size, window_size2):
        audio_segment = self.audio
        sr = self.sr
        labels = self.labels

        clip_len = librosa.get_duration(audio_segment, sr) * MILL_TO_SEC  # length in milliseconds
        start = 0
        i = 0
        num_i = int(clip_len / clip_size)       # number of clips we will create
        data = pd.DataFrame()
        leftover = 0
        leftover2 = 0

        for data_type in data_types:
            print(data_type)
            if data_type not in SUPPORTED_DATA_TYPES:
                raise ValueError("AudioFile build_dataframe(): datatype must be one of {}".format(SUPPORTED_DATA_TYPES))

        rms_full = None
        data_dict = {}

        if 'mean_full' in data_types:
            rms_full = librosa.feature.rmse(y=audio_segment, frame_length=1, hop_length=1).flatten()
            mean_full = np.mean(rms_full)
            data_dict['mean_full'] = mean_full
        if 'sd_full' in data_types:
            if rms_full is None:
                rms_full = librosa.feature.rmse(y=audio_segment, frame_length=1, hop_length=1).flatten()
            sd_full = np.std(rms_full)
            data_dict['sd_full'] = sd_full

        while start + clip_size < clip_len:
            end = start + clip_size
            new_clip = audio_segment[start:end]  # creates a .5 second clip

            surr_clip = None

            if ('fourier_surr' in data_types) or ('mean_surr' in data_types) or ('sd_surr' in data_types):
                if start - window_size < 0:
                    surr_start = 0
                    leftover = (start - window_size) * -1
                else:
                    surr_start = start - window_size
                if end + window_size + leftover > clip_len:
                    surr_end = clip_len  # we may want to change this so that it wil ALWAYS be 10 seconds
                    new_start = surr_start + (clip_len - (end + window_size + leftover))
                    if new_start >= 0:
                        surr_start = new_start
                    else:
                        surr_start = 0
                else:
                    surr_end = end + window_size + leftover
                surr_clip = audio_segment[
                            int(surr_start):int(surr_end)]  # creates 10 second clip of the surrounding audio

            surr_clip2 = None
            if ('mean_surr2' in data_types) or ('sd_surr2' in data_types) or ('mean_surr_left' in data_types) or ('sd_surr_left' in data_types):
                if start - window_size2 < 0:
                    surr_start2 = 0
                    surr_start_left = 0
                    leftover2 = (start - window_size2) * -1
                else:
                    surr_start2 = start - window_size2
                    surr_start_left = start - window_size2
                if end + window_size2 + leftover2 > clip_len:
                    surr_end2 = clip_len  # we may want to change this so that it wil ALWAYS be 10 seconds
                    new_start2 = surr_start2 + (clip_len - (end + window_size2 + leftover2))
                    if new_start2 >= 0:
                        surr_start2 = new_start2
                    else:
                        surr_start2 = 0
                else:
                    surr_end2 = end + window_size2 + leftover2
                surr_clip2 = audio_segment[
                            int(surr_start2):int(surr_end2)]  # creates 10 second clip of the surrounding audio
                surr_clip_left = audio_segment[int(surr_start_left):int(end)]

                if 'mean_surr_left' in data_types:
                    mean = AudioFile.get_rmse(surr_clip_left, 'mean')
                    data_dict['mean_surr_left'] = mean

                if 'sd_surr_left' in data_types:
                    sd = AudioFile.get_rmse(surr_clip_left, 'sd')
                    data_dict['sd_surr_left'] = sd

            if 'label' in data_types:
                seg_label = get_label(start, end, labels)  # find label for clip
                data_dict['label'] = seg_label

            if ('mean' in data_types) or ('sd' in data_types):
                rms = librosa.feature.rmse(y=new_clip, frame_length=1, hop_length=1).flatten()
                if 'mean' in data_types:
                    mean = np.mean(rms)
                    data_dict['mean'] = mean
                if 'sd' in data_types:
                    sd = np.std(rms)
                    data_dict['sd'] = sd

            if ('mean_left' in data_types) or ('sd_left' in data_types):
                if i > 0:
                    left_clip = audio_segment[start - clip_size:end - clip_size]
                else:
                    left_clip = new_clip
                if 'mean_left' in data_types:
                    mean = AudioFile.get_rmse(left_clip, 'mean')
                    data_dict['mean_left'] = mean
                if 'sd_left' in data_types:
                    sd = AudioFile.get_rmse(left_clip, 'sd')
                    data_dict['sd_left'] = sd

            if ('mean_right' in data_types) or ('sd_right' in data_types):
                if i < num_i - 1:
                    right_clip = audio_segment[start + clip_size:end + clip_size]
                else:
                    right_clip = new_clip
                if 'mean_right' in data_types:
                    mean = AudioFile.get_rmse(right_clip, 'mean')
                    data_dict['mean_right'] = mean
                if 'sd_right' in data_types:
                    sd = AudioFile.get_rmse(right_clip, 'sd')
                    data_dict['sd_right'] = sd

            if ('sd_surr' in data_types) or ('mean_surr' in data_types):
                surr_rms = librosa.feature.rms(y=surr_clip).flatten()
                if 'sd_surr' in data_types:
                    sd_surr = np.std(surr_rms)
                    data_dict['sd_surr'] = sd_surr
                if 'mean_surr' in data_types:
                    mean_surr = np.mean(surr_rms)
                    data_dict['mean_surr'] = mean_surr

            if ('sd_surr2' in data_types) or ('mean_surr2' in data_types):
                surr_rms2 = librosa.feature.rms(y=surr_clip2).flatten()
                if 'sd_surr2' in data_types:
                    sd_surr2 = np.std(surr_rms2)
                    data_dict['sd_surr2'] = sd_surr2
                if 'mean_surr2' in data_types:
                    mean_surr2 = np.mean(surr_rms2)
                    data_dict['mean_surr2'] = mean_surr2

            if 'fourier' in data_types:
                fourier = np.abs(librosa.stft(new_clip))
                # print(fourier.shape)
                fourier = fourier.flatten()
                data_dict['fourier'] = fourier

            if 'fourier_surr' in data_types:
                fourier_surr = np.abs(librosa.stft(surr_clip)).flatten()
                data_dict['fourier_surr'] = fourier_surr

            # if i % 100 == 0:
                # print("adding row ", i, " of ", num_i, " to the DataFrame")

            # at this point, all values should be None except those that we want
            data = data.append(data_dict, ignore_index=True)

            start += clip_size
            leftover = 0
            i += 1
        data_lengths = []
        columns = list(data)
        for i in columns:  # FIND LENGTH OF EACH COLUMN IN DATA
            if data[i].dtype == object:
                data_lengths.append(len(data[i][int(data[i].__len__()/2)]))
            else:
                data_lengths.append(1)
        return data_lengths, data


# builds giant dataframe of specified data for all specified wav files
class AudioData:
    def __init__(self, data_types, files=get_valid_audio_files(), clip_size=None, window_size=None, window_size2=None):
        # create new file
        self.data_types = data_types                            # an array of kinds of data for these files
        self.file_names = files
        if clip_size is None:
            self.clip_size = DEFAULT_CLIP_SIZE                  # float of the size of each clip
        else:
            self.clip_size = clip_size                          # float of the size of each clip
        if window_size is None:
            self.window_size = DEFAULT_WINDOW_SIZE              # float of the size of each surrounding window
        else:
            self.window_size = window_size                      # float of the size of each surrounding window

        if window_size2 is None:
            self.window_size2 = DEFAULT_WINDOW_SIZE2              # float of the size of each surrounding window
        else:
            self.window_size2 = window_size2                      # float of the size of each surrounding window

        list_of_objs = []
        for file in files:
            if file in get_valid_audio_files():
                x = AudioFile(file)
                list_of_objs.append(x)
            else:
                raise ValueError("AudioData _init_(): filename must be one of {}.".format(get_valid_audio_files()))
        self.files = list_of_objs                               # list of AudioFile objects
        self.data = pd.DataFrame()
        for file in self.files:
            # print('Building dataframe for {}'.format(file))
            # print(self.data_types)
            self.data_lengths, file_data = file.build_dataframe(self.data_types, self.clip_size, self.window_size,
                                                                self.window_size2)
            # data_lengths[i] = number of columns of data_types[i]
            self.data = self.data.append(file_data)
        if 'label' in self.data_types:
            self.data = self.data[self.data.label != -1]
        self.data.dropna()
        self.total_inputs = sum(self.data_lengths)

    def print(self):
        print(
            '\nAudioData Object:\nData Types: {}\nClip size: {}'
            '\nWindow size: {}\nFiles: {}\nData Lengths: {}\nTotal Inputs: {}\nData:\n{}\n'
            .format(self.data_types, self.clip_size, self.window_size,
                    self.file_names, self.data_lengths, self.total_inputs, self.data.head(3)))

    def pickle_data(self):
        curr_num = get_last_data_num() + 1
        pickle_filename = '{}data{:03d}.pickle'.format(DATA_FOLDER_PATH, curr_num)
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self, f)
        print('\nSaved as {}'.format(pickle_filename))

# TEST MEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    def add_rows(self, files):
        files_to_add = []
        skipped_files = []
        for file in files:
            if file not in get_valid_audio_files():
                raise ValueError("AudioData add_rows(): filename must be one of {}.".format(get_valid_audio_files()))
            elif file in self.file_names:
                print('{} is already included in this AudioData object. Skipping this file.'.format(file))
                skipped_files = skipped_files.append(file)
            else:
                # x = AudioFile(file)
                # list_of_objs.append(x)
                self.files = self.files.append(file)
                files_to_add = files_to_add.append(file)
        for file in files_to_add:
            print('Building dataframe for {}'.format(file))
            data_lengths, file_data = file.build_dataframe(self.data_types, self.clip_size, self.window_size)
            # data_lengths[i] = number of columns of data_types[i]
            self.data = pd.concat([self.data, file_data], ignore_index=True)
        self.data = self.data[self.data.label != -1]
        self.data.dropna()
        self.print()
        print('Skipped the following files: {}'.format(skipped_files))

# TEST MEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    def add_cols(self, data_types):
        data_to_add = []
        data_lengths = []
        for data_type in data_types:
            if data_type not in SUPPORTED_DATA_TYPES:
                raise ValueError("AudioFile add_cols(): datatype must be one of {}".format(SUPPORTED_DATA_TYPES))
            elif data_type in self.data_types:
                print('\n{} already in data_types of AudioData object. Skipping this data_type.\n'.format(data_type))
            else:
                data_to_add.append(data_type)
        data = pd.DataFrame()
        if bool(data_to_add):  # makes it this far
            for file in self.files:
                data_lengths, file_data = file.build_dataframe(data_to_add, self.clip_size, self.window_size,
                                                               self.window_size2)
                # print(file_data.head(5))
                data = data.append(file_data)
                print('\n\tdata AFTER append: \n{}\n'.format(data.head(3)))
            print(data.tail(5))  # FIX THIS: NOT ADDING NEW DATA TO OLD FILE
            self.data = pd.concat([self.data, data], axis=1)
        self.data_types = self.data_types + data_to_add
        print(data_to_add)
        print(self.data_types)
        # self.data_types.append(data_to_add)
        self.data = self.data[self.data.label != -1]
        self.data.dropna()
        self.data_lengths = self.data_lengths + data_lengths
        self.total_inputs = sum(self.data_lengths)

    # keeps only the columns of the specified data_types
    def remove_cols(self, data_types):
        self.data = self.data[data_types]
        self.data_types = data_types

    # return a prepped version of the data, splits into train/test data and scales
    def prep_for_ann(self, scaled=False, split=None):  # if split is None, that means we have no labels
        # define input dimension
        flattened_data = pd.DataFrame()
        for data_type in self.data_types:
            if data_type == 'fourier':
                fourier = self.data.fourier.apply(pd.Series)
                flattened_data.reset_index(drop=True, inplace=True)
                fourier.reset_index(drop=True, inplace=True)
                flattened_data = pd.concat((flattened_data, fourier), axis=1)
            elif data_type == 'fourier_surr':
                fourier = self.data.fourier_surr.apply(pd.Series)
                flattened_data.reset_index(drop=True, inplace=True)
                fourier.reset_index(drop=True, inplace=True)
                flattened_data = pd.concat((flattened_data, fourier), axis=1)
            elif data_type != 'label':
                curr_data = pd.DataFrame(self.data[[data_type]].values.tolist())
                flattened_data = pd.concat((flattened_data, curr_data), axis=1)

        X = flattened_data.values

        # encode class values as integers
        if split is not None:
            y = self.data[['label']].values
            encoder = LabelEncoder()
            encoded_y = encoder.fit_transform(y.ravel())
            # convert integers to dummy variables (i.e. one hot encoded)
            y = np_utils.to_categorical(encoded_y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=SEED)
            input_dim = self.total_inputs - 1

        # # Feature Scaling
        # # we don't need this because data is already between 0 and 1?
            if scaled:
                sc = StandardScaler()

                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
            return input_dim, encoder, X_train, X_test, y_train, y_test
        else:
            input_dim = self.total_inputs
            if scaled:
                sc = StandardScaler()
                X = sc.fit_transform(X)
            return input_dim, X
