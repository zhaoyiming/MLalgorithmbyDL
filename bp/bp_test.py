from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets('../mnist/', one_hot=True)
X_test, Y_test = mnist.test.images, mnist.test.labels

cnn = Sequential()
cnn.add(Dense(128, input_shape=(784,), activation="sigmoid"))
cnn.add(Dense(10, activation="sigmoid"))
cnn.load_weights("data/all.hdf5")

y_pred = cnn.predict_classes(X_test)
y_test = np.argmax(Y_test, axis=1)
np.savetxt('data/expected.txt', y_test, fmt='%01d')
np.savetxt('data/predicted.txt', y_pred, fmt='%01d')
cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
loss, accuracy = cnn.evaluate(X_test, Y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
