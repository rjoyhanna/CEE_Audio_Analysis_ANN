# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from process_data import get_data
from sklearn.utils import shuffle

# import audio file
#
# Importing the dataset
# dataset = pd.read_csv('Churn_Modelling.csv')
dataset = get_data()
dataset = shuffle(dataset)


fouriers = pd.DataFrame(dataset.iloc[:, 1].values.tolist())
surr_fouriers = pd.DataFrame(dataset.iloc[:, 4].values.tolist())
newData = np.concatenate((fouriers, surr_fouriers), axis=1)
# X = dataset.iloc[:, 3:13].values
# y = dataset.iloc[:, 13].values
# X = dataset.iloc[:, 1].values
X = newData
# X.shape(1025,1)
y = dataset.iloc[:, 2].values
# print(dataset.iloc[:,1])
# print(dataset.iloc[:,2])
# print(dataset.iloc[:,3])
# print(dataset.iloc[:,4]) # surr_fourier


# # Encoding categorical data
# # We don't need this because our data does not need to be encoded?
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X_1 = LabelEncoder()
# X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# labelencoder_X_2 = LabelEncoder()
# X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# onehotencoder = OneHotEncoder(categorical_features=[1])
# X = onehotencoder.fit_transform(X).toarray()
# X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# # Feature Scaling
# # we don't need this because data is already between 0 and 1, besides won't work since we pass a pointer to DF
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=43050, units=6, kernel_initializer="uniform"))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=10)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
# y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)