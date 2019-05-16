# Artificial Neural Network

# Part 1 - Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from process_data import get_ann_data
from sklearn.utils import shuffle
import pickle


# grab whatever type of data from whatever source you want
def configure_data(data_age, data_type):
    valid_age = {'new', 'pickle'}
    valid_type = {'mean_sd', 'fourier'}
    dataset = pd.DataFrame()
    if data_age not in valid_age:
        raise ValueError("configure_data: data_age must be one of {}." .format(valid_age))
    if data_type not in valid_type:
        raise ValueError("configure_data: data_type must be one of {}." .format(valid_type))
    if data_age == 'new':
        print("Preparing to build your new data...\n\n")
        dataset = get_ann_data(data_type)
    elif data_age == 'pickle':
        print("Opening the pickled data...\n\n")
        if data_type == 'mean_sd':
            pickle_filename = 'mean_sd.pickle'
        elif data_type == 'fourier':
            pickle_filename = 'fourier.pickle'
        else:
            pickle_filename = ''
        pickle_in = open(pickle_filename,'rb')
        dataset = pickle.load(pickle_in)
    return shuffle(dataset)


def split_data(dataset, data_type):
    # Splitting the dataset into the Training set and Test set
    if data_type == 'fourier':
        print("Extracting fourier data from the DataFrame...\n\n")
        # convert fourier results to lists so they can be passed as x values to the neural network
        # fouriers = pd.DataFrame(dataset[["fourier"]].values.tolist())
        # surr_fouriers = pd.DataFrame(dataset[["surr_fourier"]].values.tolist())
        fouriers = pd.DataFrame(dataset.iloc[:, 1].values.tolist())
        surr_fouriers = pd.DataFrame(dataset.iloc[:, 4].values.tolist())
        print("combining the fourier data...\n\n")
        newData = np.concatenate((fouriers, surr_fouriers), axis=1)
        print("defining x and y values...\n\n")
        X = newData
    elif data_type == 'mean_sd':
        print("Extracting mean/SD data from the DataFrame...\n\n")
        # convert mean/SD results to lists so they can be passed as x values to the neural network
        clips = pd.DataFrame(dataset.iloc[:, 0].values.tolist())
        sds = pd.DataFrame(dataset[['sd']].values.tolist())  # .iloc[:, 3]
        means = pd.DataFrame(dataset[['mean']].values.tolist())  # .iloc[:, 2]
        print(clips)
        print("combining the mean/SD data...\n\n")
        newData = np.concatenate((clips, sds, means), axis=1)
        print("defining x and y values...\n\n")
        X = newData
    else:
        X = pd.DataFrame()
    y = dataset[['label']].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    # # Feature Scaling
    # # we don't need this because data is already between 0 and 1
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    print(X)
    return X_train, X_test, y_train, y_test


def run_ann(data_age, data_type, num_units, num_hidden_layers):
    dataset = configure_data(data_age, data_type)
    X_train, X_test, y_train, y_test = split_data(dataset, data_type)
    if data_type == 'fourier':
        num_inputs = 8200
    elif data_type == 'mean_sd':
        num_inputs = 502
    else:
        num_inputs = 0
    # Part 2 - Now let's make the ANN!

    # Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(activation="relu", input_dim=num_inputs, units=num_units, kernel_initializer="uniform"))

    # Adding the second hidden layer
    classifier.add(Dense(activation="relu", units=num_units, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=num_units, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=num_units, kernel_initializer="uniform"))

    # Adding the output layer
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

    # Compiling the ANN
    # classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # optimizer = optimizer_rmsprop(), loss = loss_binary_crossentropy, metrics = metric_binary_accuracy)
    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fitting the ANN to the Training set
    history = classifier.fit(X_train, y_train, batch_size=128, epochs=20)

    # Part 3 - Making the predictions and evaluating the model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    # y_pred = (y_pred > 0.5)

    plt.plot(history.history['acc'])
    plt.show()

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)


run_ann('pickle', 'fourier', 20, 3)


# # Encoding categorical data
# # We don't need this because our data does not need to be encoded (numerical already)?
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X_1 = LabelEncoder()
# X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# labelencoder_X_2 = LabelEncoder()
# X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# onehotencoder = OneHotEncoder(categorical_features=[1])
# X = onehotencoder.fit_transform(X).toarray()
# X = X[:, 1:]

