import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

data = pd.read_csv("abalone.csv")
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
data = np.array(data)
X = data[:,:7]
y = data[:,8]

idx = 1
for i in range(1, 100):
    reg = RandomForestRegressor(n_estimators=i, random_state=1)
    reg.fit(X, y)
    kf = KFold(n_splits=5, shuffle = True, random_state= 42)
    score = cross_val_score(reg, X, y, cv=kf, scoring='r2')
    print(score.mean(), idx)
    idx += 1



