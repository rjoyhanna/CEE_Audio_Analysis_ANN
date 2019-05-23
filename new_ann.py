# Train model and make predictions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from process_data import get_ann_data
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import pickle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


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
    return dataset


def run_ann(data_age, data_type, num_units, num_hidden_layers, num_epochs, batch_size, test_train_split):
    # Splitting the dataset into the Training set and Test set
    dataset = configure_data(data_age, data_type)

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
        print("combining the mean/SD data...\n\n")
        newData = np.concatenate((clips, sds, means), axis=1)
        print("defining x and y values...\n\n")
        X = newData
    else:
        X = pd.DataFrame()
    y = dataset[['label']].values
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    print('classes:\n', encoder.classes_)
    encoded_y = encoder.transform(y)
    print(encoded_y)
    i = 0
    num_labelled_0 = 0
    num_labelled_1 = 0
    num_labelled_2 = 0
    num_labelled_3 = 0
    while i < len(encoded_y):
        if encoded_y[i] == 0:
            num_labelled_0 += 1
        elif encoded_y[i] == 1:
            num_labelled_1 += 1
        elif encoded_y[i] == 2:
            num_labelled_2 += 1
        elif encoded_y[i] == 3:
            num_labelled_3 += 1
        i += 1
    print('num labelled 0: ', num_labelled_0, '\n')
    print('num labelled 1: ', num_labelled_1, '\n')
    print('num labelled 2: ', num_labelled_2, '\n')
    print('num labelled 3: ', num_labelled_3, '\n')

    # convert integers to dummy variables (i.e. one hot encoded)
    y = np_utils.to_categorical(encoded_y)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_train_split, random_state=seed)

    if data_type == 'fourier':
        num_inputs = 8200
    elif data_type == 'mean_sd':
        num_inputs = 1027
    else:
        num_inputs = 0

    # # Feature Scaling
    # # we don't need this because data is already between 0 and 1
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    data_to_pickle = sc
    pickle_filename = 'scalar.pickle'
    with open(pickle_filename, 'wb') as f:
        pickle.dump(data_to_pickle, f)

    # Initialising the ANN
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    # classifier.add(Dense(activation="relu", input_dim=num_inputs, units=num_units, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", input_dim=num_inputs, units=num_units))
    # Adding the other hidden layers
    i = 0
    while i < num_hidden_layers:
        classifier.add(Dense(activation="relu", units=num_units, kernel_initializer="uniform"))
        i += 1
    # Adding the output layer
    # classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.add(Dense(activation="softmax", units=4, kernel_initializer="uniform"))
    # Compiling the ANN
    classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fitting the ANN to the Training set
    class_weight = {0: 1./num_labelled_0,
                    1: 1./num_labelled_1,
                    2: 1./num_labelled_2,
                    3: 1./num_labelled_3}
    hist = classifier.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, class_weight=class_weight)

    # Part 3 - Making the predictions and evaluating the model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1, out=None)

    y_test = np.argmax(y_test, axis=1, out=None)

    fig1 = plt.figure(1)
    plt.plot(hist.history["loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    fig1.show()

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    labels = [0, 1, 2, 3]
    cm = confusion_matrix(y_test, y_pred, labels)
    # print(cm)
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig2.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return classifier


# def run_ann(data_age, data_type, num_units, num_hidden_layers, num_epochs, batch_size, test_train_split):

run_ann('pickle', 'mean_sd', 20, 2, 20, 128, .4)#.save('mean_sd_ann.h5')

# estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
# X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=seed)
# estimator.fit(X_train, Y_train)
# predictions = estimator.predict(X_test)
# print(predictions)
# print(encoder.inverse_transform(predictions))