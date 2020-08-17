from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Dense
from keras.models import Sequential

mnist = input_data.read_data_sets('../mnist/', one_hot=True)
X_train,Y_train = mnist.train.images, mnist.train.labels


cnn = Sequential()
cnn.add(Dense(10, input_shape=(784,), activation="softmax"))


cnn.compile(loss="categorical_crossentropy", optimizer="sgd",metrics=['accuracy'])
cnn.fit(X_train, Y_train, nb_epoch=10, validation_split=0.2)
cnn.save("data/lr.hdf5")
