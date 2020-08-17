from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Activation
import numpy as np

mnist = input_data.read_data_sets('../mnist/', one_hot=True)
X_test, Y_test = mnist.test.images, mnist.test.labels

X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
model = Sequential()
model.add(LSTM(64,input_dim=784, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(64,input_dim=784, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('softmax'))

model.load_weights("data/lstm.hdf5")

y_pred = model.predict_classes(X_test)
y_test = np.argmax(Y_test, axis=1)
np.savetxt('data/expected.txt', y_test, fmt='%01d')
np.savetxt('data/predicted.txt', y_pred, fmt='%01d')
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

loss, accuracy = model.evaluate(X_test, Y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
