from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense, Dropout, Activation
import numpy as np

mnist = input_data.read_data_sets('../mnist/', one_hot=True)
X_train,Y_train = mnist.train.images, mnist.train.labels

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))



model = Sequential()
model.add(GRU(64,input_dim=784, return_sequences=True))
model.add(Dropout(0.1))
model.add(GRU(64,input_dim=784, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('softmax'))


model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])
model.fit(X_train, Y_train, nb_epoch=10, validation_split=0.2)
model.save("data/gru.hdf5")
