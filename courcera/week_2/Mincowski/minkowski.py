import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data = datasets.load_boston()
X = data.data
y = data.target
X = preprocessing.scale(X)


kf = KFold(n_splits=5, shuffle = True, random_state= 42)
counter = np.linspace(1.0, 10.0, num=200)

for i in counter:   
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', p=i, metric='minkowski', metric_params=None)
#    neigh.fit(X, y) 
    score = cross_val_score(neigh, X, y, None, scoring='neg_mean_squared_error', cv=kf)
    mean = score.mean()
    print(mean)