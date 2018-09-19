import pandas as p
import numpy as np
from sklearn.model_selection import cross_val_score
#data = open("wine.data", "r")
#printdata = str(data.read())
#print(printdata)

#data = p.read_csv("wine.csv")
#newlist = printdata.split("\n")
#for row in newlist:
#     print(row)
#data.close()

dat = np.array(p.read_csv('wine.data.csv', header = None))
X = dat[:,1:]
y = dat[:,0]

#y = data['A']
#X = data[['B','C','D','E','F','G','H','I','G','K','L','M','N']]

from sklearn import preprocessing
X = preprocessing.scale(X)

#print(type(X))
#print(type(y))
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle = True, random_state= 42)


#for train_index, test_index in kf.split(X):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]


from sklearn.neighbors import KNeighborsClassifier

for i in range(1, 50):    
    clf = KNeighborsClassifier(i)
    clf.fit(X, y)
    score = cross_val_score(clf, X, y, None, None, kf)
    print(score.mean())



#predictions = clf.predict(X_test)
#from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test, predictions))