import pandas as p
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

dat = np.array(p.read_csv('wine.data.csv', header = None))
X = dat[:,1:]
y = dat[:,0]
X = preprocessing.scale(X)
print(X)

kf = KFold(n_splits=5, shuffle = True, random_state= 42)

for i in range(1, 50):    
    clf = KNeighborsClassifier(i)
    clf.fit(X, y)
    score = cross_val_score(clf, X, y, None, None, kf)
    print(score.mean())
