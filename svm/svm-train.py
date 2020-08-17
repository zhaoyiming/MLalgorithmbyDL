from keras.regularizers import l2
from keras.models import Sequential
from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Dense, Activation
import tensorflow as tf

mnist = input_data.read_data_sets('../mnist/', one_hot=True)
X_train,Y_train = mnist.train.images,mnist.train.labels


cnn = Sequential()
cnn.add(Dense(10, input_shape=(784,), W_regularizer=l2(0.00001)))
cnn.add(Activation('softmax'))

#  模拟svm
cnn.compile(loss='squared_hinge',
              optimizer='adadelta',
              metrics=['accuracy'])
cnn.fit(X_train, Y_train, nb_epoch=10, validation_split=0.2)
cnn.save("data/svm.hdf5")



