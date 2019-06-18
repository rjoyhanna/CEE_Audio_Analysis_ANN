import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

DATA_FOLDER_PATH = 'C:\\Users\\rjoyh\\Desktop\\machine learning\\data\\'
SEED = 7


# self.filename is a string of the form 'mean_sd1.pickle' or 'fourier1.pickle'
# self.dataframe is a dataframe of the raw data given in the pickled file
# self.type is a string either 'fourier' or 'mean_sd'
# self.rows is the number of half second clips of data
# self.cols is the number of types of input data (like mean, sd, fourier, etc.)
# self.input_dim is the number of inputs to the ann, it breaks fourier data into individual inputs
# self.scaled is a boolean: True means the data is scaled
# self.encoder is an encoder object used on this data
# self.sc is a scalar object used on this data
# self.X_train is a dataframe
# self.X_test is a dataframe
# self.y_train is a dataframe
# self.y_test is a dataframe
class Data:
    def __init__(self, filename, split, scaled):
        self.filename = filename
        self.split = split
        # r=root, d=directories, f = files
        valid_files = []
        for r, d, f in os.walk(DATA_FOLDER_PATH):
            valid_files = f
        # in the future, create an option to add new data
        if filename not in valid_files:
            raise ValueError("get_data: data must be one of {}.".format(valid_files))
        pickle_in = open(DATA_FOLDER_PATH + filename, 'rb')

        self.dataframe = pickle.load(pickle_in)
        self.type = filename[0:7]
        self.rows = self.dataframe.shape[0]  # gives number of row count
        self.cols = self.dataframe.shape[1]  # gives number of col count
        if self.type == 'fourier':
            self.input_dim = 8200
        elif self.type == 'mean_sd':
            if self.cols == 4:
                self.input_dim = 1027
            elif self.cols == 8:
                self.input_dim = 1031
            else:
                raise ValueError("get_data: columns of {} not accounted for.".format(self.cols))
        self.scaled = scaled
        self.encoder, self.sc, self.X_train, self.X_test, self.y_train, self.y_test = self.merge_data()
        self.print_vals()

    # flatten fourier data into individual values
    # concatenate that with the other data so we have all columns of ints
    # encode, scale data
    # return test/train split
    def merge_data(self):
        if self.type == 'fourier':
            print("Extracting fourier data from the DataFrame...\n\n")
            # convert fourier results to lists so they can be passed as x values to the neural network
            fouriers = pd.DataFrame(self.dataframe.iloc[:, 1].values.tolist())
            surr_fouriers = pd.DataFrame(self.dataframe.iloc[:, 4].values.tolist())
            print("combining the fourier data...\n\n")
            newData = np.concatenate((fouriers, surr_fouriers), axis=1)
            print("defining x and y values...\n\n")
            X = newData
        elif self.type == 'mean_sd':
            print("Extracting mean/SD data from the DataFrame...\n\n")
            # convert mean/SD results to lists so they can be passed as x values to the neural network
            clips = pd.DataFrame(self.dataframe.iloc[:, 0].values.tolist())
            print("combining the mean/SD data...\n\n")
            if self.cols == 8:
                sds = pd.DataFrame(self.dataframe[['sd']].values.tolist())  # .iloc[:, 3]
                means_surr = pd.DataFrame(self.dataframe[['mean_surr']].values.tolist())
                means = pd.DataFrame(self.dataframe[['mean']].values.tolist())  # .iloc[:, 2]
                sds_surr = pd.DataFrame(self.dataframe[['sd_surr']].values.tolist())
                sds_full = pd.DataFrame(self.dataframe[['sd_full']].values.tolist())
                means_full = pd.DataFrame(self.dataframe[['mean_full']].values.tolist())
                X = np.concatenate((clips, sds, means, means_surr, sds_surr, sds_full, means_full), axis=1)
            else:
                sds = pd.DataFrame(self.dataframe[['sd']].values.tolist())  # .iloc[:, 3]
                means_surr = pd.DataFrame(self.dataframe[['mean']].values.tolist())
                X = np.concatenate((clips, sds, means_surr), axis=1)
        else:
            X = pd.DataFrame()
        y = self.dataframe[['label']].values
        # encode class values as integers
        encoder = LabelEncoder()
        encoded_y = encoder.fit_transform(y)

        # convert integers to dummy variables (i.e. one hot encoded)
        y = np_utils.to_categorical(encoded_y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split, random_state=SEED)

        # # Feature Scaling
        # # we don't need this because data is already between 0 and 1?
        if self.scaled:
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()

            X_train_fouriers = sc.fit_transform(X_train[:, 0: 1026])
            X_test_fouriers = sc.transform(X_test[:, 0: 1026])
            X_train[:, 0: 1026] = X_train_fouriers
            X_test[:, 0: 1026] = X_test_fouriers
        else:
            sc = None
        return encoder, sc, X_train, X_test, y_train, y_test

    def print_vals(self):
        print('\nData Object from {}'.format(self.filename))
        print(' * data type: {}'.format(self.type))
        print(' * input dimension: {}'.format(self.input_dim))
