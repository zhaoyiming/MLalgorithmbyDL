from sklearn.externals import joblib

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score

mnist = input_data.read_data_sets('../mnist/')
X_train,Y_train = mnist.train.images,mnist.train.labels

gnb = GaussianNB()
gnb=MultinomialNB()
gnb.fit(X_train, Y_train)
joblib.dump(gnb,  "data/nb.pkl")
predict_labels = gnb.predict(X_train)
Accuracy = accuracy_score(Y_train, predict_labels)
print(Accuracy)

