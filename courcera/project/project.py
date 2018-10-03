import numpy as np
import pandas as pd

data = pd.read_csv("features.csv", index_col='match_id')
data = data.sample(n=10000)

y = np.array(data.radiant_win)
#y = y.reshape((y.shape[0], 1))

# Deleting unused features
features = data.drop(["duration", "radiant_win", "tower_status_radiant",
                          "tower_status_dire", "barracks_status_radiant",
                          "barracks_status_dire", "start_time"], axis=1 )



# Deleting NotNaN or Imputing missing values
# 1st Deleting NotNaN
#features = features.dropna()
#y = y.dropna()

# 2nd Imputing missing values
from sklearn.preprocessing import Imputer
columns = features.axes[1]
rows = features.axes[0]
imr = Imputer(missing_values="NaN", strategy="mean", axis=0)
imr.fit(features)
features = imr.transform(features.values)
features = pd.DataFrame(features, columns=columns, index=rows)

# One-hot
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder(categorical_features=[0])
#features = ohe.fit_transform(features).toarray()
#print(features.shape)

#print(features)

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle = True, random_state= 42)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

#for i in range(10, 40, 5):
#learning_list = [1, 0.5, 0.3, 0.2, 0.1]
#
#for i in learning_list:
clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.2, max_depth=2)
clf.fit(features, y)
score = cross_val_score(clf, features, y, cv=kf)
print(score.mean())
pred = clf.predict_proba(features)[:, 1]
print(pred.mean())
#for i, name in zip(clf.feature_importances_, columns):
#    if i != 0.0:
#        print("{0}:{1}".format(name, i))

#print(clf.feature_importances_)


#print(features.isnull().sum())

#print(features.shape)