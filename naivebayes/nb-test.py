from sklearn.externals import joblib

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

mnist = input_data.read_data_sets('../mnist/')
X_test,Y_test = mnist.test.images,mnist.test.labels

clf=joblib.load("data/nb.pkl")
predict_labels = clf.predict(X_test)
Accuracy = accuracy_score(Y_test, predict_labels)
print(Accuracy)

