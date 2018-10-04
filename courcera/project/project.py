import numpy as np
import pandas as pd
import time
import datetime

data = pd.read_csv("features.csv", index_col='match_id')

'''
Random n samples
'''
#data = data.sample(n=1000)

y = np.array(data.radiant_win)
#y = y.reshape((y.shape[0], 1))


'''
Deleting unused features
'''
features = data.drop(["duration", "radiant_win", "tower_status_radiant",
                          "tower_status_dire", "barracks_status_radiant",
                          "barracks_status_dire", "start_time"], axis=1 )


'''
''' 
'''
                    Deleting NaN or Imputing missing values
*******************************************************************************
1st option - Deleting NaN
''' 
#features = features.dropna()
#y = y.dropna()
'''
''' 
'''
2nd option - Imputing missing values
''' 
#from sklearn.preprocessing import Imputer
#columns = features.axes[1]
#rows = features.axes[0]
#imr = Imputer(missing_values="NaN", strategy="mean", axis=0)
#imr.fit(features)
#features = imr.transform(features.values)
#features = pd.DataFrame(features, columns=columns, index=rows)
'''
''' 
'''
3rd option - Changing missing values to zeroes 
''' 
features = features.fillna(0)



from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle = True, random_state= 42)
from sklearn.model_selection import cross_val_score

'''
                            Gradient Boosting
*******************************************************************************
''' 

from sklearn.ensemble import GradientBoostingClassifier

'''
Choosing the best learning rate
''' 
#for i in range(10, 40, 5):
#learning_list = [1, 0.5, 0.3, 0.2, 0.1]
#for i in learning_list:
#    clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.2, max_depth=2)
#    clf.fit(features, y)
#    score = cross_val_score(clf, features, y, cv=kf)
#    print(score.mean())
'''
GradientBoostingClassifier
''' 
start_time = datetime.datetime.now()
clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, max_depth=2)
score = cross_val_score(clf, features, y, cv=kf, scoring='roc_auc')
print("Gradient boosting time execution:", datetime.datetime.now() - start_time)
print("Gradient boosting AUC-ROC score:", score.mean())
'''
Feature importances
''' 
#for i, name in zip(clf.feature_importances_, columns):
#    if i != 0.0:
#        print("{0}:{1}".format(name, i))

'''
                            Logistic regression
*******************************************************************************
'''
'''
Changing categorical features

1st option - One-hot
''' 
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder(categorical_features=[0, 1, 9, 17, 25, 33, 41, 49, 57, 65, 73])
#features = ohe.fit_transform(features).toarray()
'''
2nd option - Deleting categorical features
''' 
#features = features.drop(["lobby_type", "r1_hero", "r2_hero", "r3_hero",
#                          "r4_hero", "r5_hero", "d1_hero", "d2_hero",
#                          "d3_hero", "d4_hero", "d5_hero"], axis=1)

'''
3rd option - Bag of words
''' 
unique_heroes = np.count_nonzero(pd.unique(features.r1_hero))
X_pick = np.zeros((features.shape[0], 112))
for i, match_id in enumerate(features.index):
    for p in range(5):
        X_pick[i, features.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, features.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1        

features = np.append(np.array(features), X_pick, axis=1)

'''
Feature scaling
''' 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
features = scaler.fit_transform(features)


from sklearn.linear_model import LogisticRegression
'''
Choosing the best regularization (C)
''' 
#c_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
#for i in c_list:
#    clf = LogisticRegression(C=i, random_state=42)
#    score = cross_val_score(clf, features, y, cv=kf)
#    print(score.mean())

'''
Logistic Regression Classifier
''' 
start_time = datetime.datetime.now()
clf = LogisticRegression(C=0.001, random_state=42)
score = cross_val_score(clf, features, y, cv=kf, scoring='roc_auc')
print("Logistic regression time execution:", datetime.datetime.now() - start_time)
print("Logistic regression AUC-ROC score:", score.mean())
clf.fit(features, y)
pred = clf.predict_proba(features)[:, 1]
print("Min prediction:", min(pred))
print("Max prediction:", max(pred))

















