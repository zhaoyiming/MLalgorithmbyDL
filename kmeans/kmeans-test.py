from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Model, Sequential  # 泛型模型
from keras.layers import Dense, Input
import numpy as np

mnist = input_data.read_data_sets('../mnist/', one_hot=True)
X_test, Y_test = mnist.test.images, mnist.test.labels
X_train,Y_train = mnist.train.images, mnist.train.labels


input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
autoencoder = Model(input_img, decoded)

autoencoder.load_weights("data/autocoder.hdf5")

y_pred = autoencoder.predict(X_test)

np.savetxt('data/expected.txt', X_test)
np.savetxt('data/predicted.txt', y_pred)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])


loss, accuracy = autoencoder.evaluate(X_test, X_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
