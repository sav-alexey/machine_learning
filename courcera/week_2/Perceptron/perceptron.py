import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data1 = np.array(pd.read_csv('perceptron-test.csv', header = None))
data2 = np.array(pd.read_csv('perceptron-train.csv', header = None))

X_train = data2[:,1:]
y_train = data2[:,0]

X_test = data1[:,1:]
y_test = data1[:,0]

scaler = StandardScaler()
X_train_tran = scaler.fit_transform(X_train, y_train)
X_test_tran = scaler.transform(X_test, y_test)

#print(X_train)
#print(y_train)

clf = Perceptron(random_state=241, max_iter=5, tol=None)
clf.fit(X_train_tran, y_train)
predictions = clf.predict(X_test_tran)
score1 = accuracy_score(y_test, predictions)

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
score2 = accuracy_score(y_test, predictions)
print(score1, score2)