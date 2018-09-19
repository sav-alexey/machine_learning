import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


data = np.array(pd.read_csv('svm-data.csv', header = None))

#X = data[:,1:3]
#y = data[:,0:1]

X = data[:,1:]
y = data[:,0]

clf = SVC( C = 100000, kernel='linear', random_state=241)
clf.fit(X, y)

lis = clf.support_


print(clf.support_+1)