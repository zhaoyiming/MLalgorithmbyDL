import numpy as np

np.random.seed(1337)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Model, Sequential  # 泛型模型
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('../mnist/', one_hot=True)
X_train,Y_train = mnist.train.images, mnist.train.labels
X_test, Y_test = mnist.test.images, mnist.test.labels

input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.fit(X_train, X_train,
                nb_epoch=10,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))
autoencoder.save("data/autocoder.hdf5")