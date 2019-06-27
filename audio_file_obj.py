from file_helper import add_txt
from file_helper import add_wav
from file_helper import get_valid_audio_files
from file_helper import get_last_data_num
from file_helper import DATA_FOLDER_PATH
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import pandas as pd
import librosa
import numpy as np
import pickle

SUPPORTED_DATA_TYPES = ['fourier', 'fourier_surr', 'mean', 'sd', 'mean_surr', 'sd_surr', 'sd_full', 'mean_full', 'label']
DEFAULT_CLIP_SIZE = 500  # half second total
DEFAULT_WINDOW_SIZE = 2500  # 5 seconds total
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


# filename should not include an extension
class AudioFile:
    def __init__(self, filename):
        self.name = filename
        # create option for building dataframe with no labels
        self.labels = pd.read_csv(add_txt(self.name), delimiter="\t", names=["start", "end", "label"])
        self.audio, self.sr = librosa.load(add_wav(filename))

    def alter_labels(self, accuracy=None):
        if accuracy is None:
            last_label = self.labels.shape[0]
            for i in range(0, last_label - 1):
                if self.labels.loc[i, 'end'] != self.labels.loc[i+1, 'start']:
                    print('\nInconsistencies in row {}\n'.format(i))
                    difference = self.labels.loc[i+1, 'start'] - self.labels.loc[i, 'end']
                    print('difference: {}\n'.format(difference))

    def build_dataframe(self, data_types, clip_size, window_size):
        audio_segment = self.audio
        sr = self.sr
        labels = self.labels

        clip_len = librosa.get_duration(audio_segment, sr) * MILL_TO_SEC  # length in milliseconds
        start = 0
        i = 0
        num_i = int(clip_len / clip_size)
        data = pd.DataFrame()
        leftover = 0

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

            if ('sd_surr' in data_types) or ('mean_surr' in data_types):
                surr_rms = librosa.feature.rms(y=surr_clip).flatten()
                if 'sd_surr' in data_types:
                    sd_surr = np.std(surr_rms)
                    data_dict['sd_surr'] = sd_surr
                if 'mean_surr' in data_types:
                    mean_surr = np.mean(surr_rms)
                    data_dict['mean_surr'] = mean_surr

            if 'fourier' in data_types:
                fourier = np.abs(librosa.stft(new_clip)).flatten()
                data_dict['fourier'] = fourier

            if 'fourier_surr' in data_types:
                fourier_surr = np.abs(librosa.stft(surr_clip)).flatten()
                data_dict['fourier_surr'] = fourier_surr

            if i % 100 == 0:
                print("adding row ", i, " of ", num_i, " to the DataFrame")

            # at this point, all values should be None except those that we want
            data = data.append(data_dict, ignore_index=True)

            start += clip_size
            leftover = 0
            i += 1
        # print(data[["label", "sd", "mean"]])
        # from scipy.stats import zscore
        # print(data[['label','sd','mean']])
        # data[['sd','mean']] = data[["sd", 'mean']].apply(zscore)  # data standardization
        # print(data[['label','sd','mean']])
        data_lengths = []
        columns = list(data)
        for i in columns:  # FIND LENGTH OF EACH COLUMN IN DATA
            if data[i].dtype == object:
                data_lengths.append(len(data[i][int(data[i].__len__()/2)]))
            else:
                data_lengths.append(1)
        return data_lengths, data


class AudioData:
    def __init__(self, data_types, files=get_valid_audio_files(), clip_size=None, window_size=None):
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
            print('Building dataframe for {}'.format(file))
            print(self.data_types)
            self.data_lengths, file_data = file.build_dataframe(self.data_types, self.clip_size, self.window_size)
            # data_lengths[i] = number of columns of data_types[i]
            self.data = self.data.append(file_data)
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
                data_lengths, file_data = file.build_dataframe(data_to_add, self.clip_size, self.window_size)
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

    def prep_for_ann(self, scaled=False, split=None):  # if split is None, that means we have no labels
        # define input dimension
        flattened_data = pd.DataFrame()
        for data_type in self.data_types:
            if data_type == 'fourier':
                # curr_data = pd.DataFrame(self.data[[data_type]].values.tolist())
                fourier = self.data.fourier.apply(pd.Series)
                # print('\n\tcurr data: \n{}'.format(curr_data.head(5)))
                # print('\n\tcurr data: \n{}'.format(flattened_data))

                flattened_data.reset_index(drop=True, inplace=True)
                fourier.reset_index(drop=True, inplace=True)
                flattened_data = pd.concat((flattened_data, fourier), axis=1)
                # print('\n\tThis is {} data:\n{}\n'.format(data_type, curr_data))
                # deal with
            elif data_type == 'fourier_surr':
                # curr_data = pd.DataFrame(self.data[[data_type]].values.tolist())
                fourier = self.data.fourier_surr.apply(pd.Series)

                flattened_data.reset_index(drop=True, inplace=True)
                fourier.reset_index(drop=True, inplace=True)
                flattened_data = pd.concat((flattened_data, fourier), axis=1)
                # print('\n\tThis is {} data:\n{}\n'.format(data_type, curr_data))
                # deal with
            elif data_type != 'label':
                curr_data = pd.DataFrame(self.data[[data_type]].values.tolist())
                # print('\n\tThis is {} data:\n{}\n'.format(data_type,curr_data))
                # flattened_data = np.concatenate((flattened_data, curr_data), axis=1)
                # flattened_data.reset_index(drop=True, inplace=True)
                # curr_data.reset_index(drop=True, inplace=True)
                # print(type(flattened_data))
                # print(type(curr_data))
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
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()

                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
            return input_dim, encoder, X_train, X_test, y_train, y_test
        else:
            # print('split is {} inside prep_for_ann'.format(split))
            input_dim = self.total_inputs
            if scaled:
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                X = sc.fit_transform(X)
            return input_dim, X


data = AudioData(['mean_surr', 'sd', 'label'], clip_size=500, window_size=7250)
data.print()
data.pickle_data()
