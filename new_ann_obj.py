import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.models import Sequential
from keras.layers import Dense
from file_helper import open_pickled_file
import datetime


class ANN:
    # initializer
    # data is a filename for now
    # ANN will be compiled with data ready to use
    # can be used on different data, but it must have same dimensions
    def __init__(self, data, num_units, num_hidden_layers, num_epochs, batch_size, test_train_split, trial_num):
        self.num_units = num_units
        self.num_hidden_layers = num_hidden_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.test_train_split = test_train_split
        # print('split is {} inside _init_()'.format(self.test_train_split))
        self.trial_num = trial_num
        self.filename = data
        raw_data = open_pickled_file(data)
        input_dim, encoder, X_train, X_test, y_train, y_test = raw_data.prep_for_ann(split=self.test_train_split)
        self.input_dim = input_dim
        self.encoder = encoder
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.classifier = self.create()
        self.hist = self.train()
        self.y_pred = self.classifier.predict(self.X_test)

    def create(self):
        # Initialising the ANN
        classifier = Sequential()
        # Adding the input layer and the first hidden layer
        # classifier.add(Dense(activation="relu", input_dim=num_inputs, units=num_units, kernel_initializer="uniform"))
        classifier.add(Dense(activation="relu", input_dim=self.input_dim, units=self.num_units))
        # Adding the other hidden layers
        i = 0
        while i < self.num_hidden_layers:
            classifier.add(Dense(activation="relu", units=self.num_units, kernel_initializer="uniform"))
            i += 1
        # Adding the output layer
        # classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
        classifier.add(Dense(activation="softmax", units=4, kernel_initializer="uniform"))
        # Compiling the ANN
        classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return classifier

    # train ANN on given data
    def train(self):
        i = 0
        num_labelled_0 = 0
        num_labelled_1 = 0
        num_labelled_2 = 0
        num_labelled_3 = 0
        while i < len(self.y_train):
            curr_label = np.argmax(self.y_train[i])
            if curr_label == 0:
                num_labelled_0 += 1
            elif curr_label == 1:
                num_labelled_1 += 1
            elif curr_label == 2:
                num_labelled_2 += 1
            elif curr_label == 3:
                num_labelled_3 += 1
            i += 1
        i = 0
        while i < len(self.y_test):
            curr_label = np.argmax(self.y_test[i])
            if curr_label == 0:
                num_labelled_0 += 1
            elif curr_label == 1:
                num_labelled_1 += 1
            elif curr_label == 2:
                num_labelled_2 += 1
            elif curr_label == 3:
                num_labelled_3 += 1
            i += 1
        # Fitting the ANN to the Training set
        class_weight = {0: 1. / num_labelled_0,
                        1: 1. / num_labelled_1,
                        2: 1. / num_labelled_2,
                        3: 1. / num_labelled_3}

        hist = self.classifier.fit(self.X_train, self.y_train, batch_size=self.batch_size,
                                   epochs=self.num_epochs, class_weight=class_weight)
        return hist

    def create_graphs(self, save=True):
        y_pred = np.argmax(self.y_pred, axis=1, out=None)
        y_test = np.argmax(self.y_test, axis=1, out=None)

        import datetime
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d %H:%M")
        fig1 = plt.figure(1)
        plt.plot(self.hist.history["loss"])
        plt.title("ANN Loss: " + date)
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plot_acc = "{:.2f}%".format((self.hist.history['acc'][len(self.hist.history['acc']) - 1]) * 100)
        text = '{}\ndata: {}, nodes: {}, layers: {}\nepochs: {}, batch size: {}, split: {}, data: {}\naccuracy: {}'\
            .format(date, self.filename, self.num_units, self.num_hidden_layers, self.num_epochs, self.batch_size,
                    self.test_train_split, len(self.y_test) + len(self.y_train), plot_acc)
        text_y = (max(self.hist.history["loss"]) + min(self.hist.history["loss"])) / 2
        text_x = self.num_epochs / 2
        plt.text(text_x, text_y, text, horizontalalignment='center')
        plt.legend(["train", "val"], loc="upper left")
        if save:
            fig1.savefig('trials' + str(self.trial_num) +
                         '/loss/{}_{}loss.png'.format(plot_acc, now.strftime("%Y-%m-%d_%H-%M-%S")))
            fig1.clf()
        else:
            fig1.show()
        # fig1.clf()

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        labels = self.encoder.classes_.astype(int)
        print(labels)
        # labels = [0, 1, 2, 3]
        # labels = encoder.transform(sorted(encoder.classes_))
        y_test = self.encoder.inverse_transform(y_test).astype(int)
        # y_pred = self.data.encoder.inverse_transform(y_pred).astype(int)
        print(y_test)
        print(y_pred)

        cm = confusion_matrix(y_test, y_pred, labels)

        # print(cm)
        fig2 = plt.figure(2)
        ax = fig2.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title("ANN Confusion Matrix: " + date)
        plt.text(1.5, 0.5, text, horizontalalignment='center', color='white')
        fig2.colorbar(cax)
        ax.set_xticklabels([''] + list(labels))
        ax.set_yticklabels([''] + list(labels))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if save:
            plt.savefig(
                'trials' + str(self.trial_num) +
                '/confusion/{}_{}con.png'.format(plot_acc, now.strftime("%Y-%m-%d_%H-%M-%S")))
            plt.clf()
        else:
            plt.show()
        i = 0
        num_labelled_0 = 0
        num_labelled_1 = 0
        num_labelled_2 = 0
        num_labelled_3 = 0
        while i < len(y_pred):
            curr_label = y_pred[i]
            if curr_label == 0:
                num_labelled_0 += 1
            elif curr_label == 1:
                num_labelled_1 += 1
            elif curr_label == 2:
                num_labelled_2 += 1
            elif curr_label == 3:
                num_labelled_3 += 1
            i += 1
        print(num_labelled_0)
        print(num_labelled_1)
        print(num_labelled_2)
        print(num_labelled_3)
        return num_labelled_0, num_labelled_1, num_labelled_2, num_labelled_3

    def save_obj(self):
        now = datetime.datetime.now()
        plot_acc = "{:.2f}%".format((self.hist.history['acc'][len(self.hist.history['acc']) - 1]) * 100)
        filename = 'trials{}/anns/{}_{}ann.pickle'.format(self.trial_num, plot_acc, now.strftime("%Y-%m-%d_%H-%M-%S"))
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
