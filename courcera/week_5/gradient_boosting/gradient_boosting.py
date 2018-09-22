import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss

data = np.array(pd.read_csv("gbm-data.csv"))
X = data[:,1:]
y = data[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

scores_train = []
scores_test = []
#learning_list = [1, 0.5, 0.3, 0.2, 0.1]
#for i in learning_list:
#    clf = GradientBoostingClassifier(n_estimators=250, verbose=False, random_state=241, learning_rate=i)
#    clf.fit(X_train, y_train)
#    for i, predict in enumerate(clf.staged_decision_function(X_test)):
#        scores_test.append(log_loss(y_test, 1 / (1 + np.exp(-predict))))
#    for i, predict in enumerate(clf.staged_decision_function(X_train)):
#        scores_train.append(log_loss(y_train, 1 / (1 + np.exp(-predict))))
#        
#        
#plt.figure()
#plt.plot(scores_test, 'r', linewidth=2)
#plt.plot(scores_train, 'g', linewidth=2)
#plt.legend(['test', 'train'])

#clf = GradientBoostingClassifier(n_estimators=250, verbose=False, random_state=241, learning_rate=0.2)
#clf.fit(X_train, y_train)
#clf.staged_predict_proba(X_train)
#clf.staged_predict_proba(X_test)  
#for i, predict in enumerate(clf.staged_decision_function(X_test)):
#    scores_test.append(log_loss(y_test, 1 / (1 + np.exp(-predict))))

clf = RandomForestClassifier(n_estimators=36, random_state=241)
clf.fit(X_train, y_train)
#clf.staged_predict_proba(X_train)
#clf.staged_predict_proba(X_test) 
#for i, predict in enumerate(clf.predict_proba(X_test)):
#    scores_test.append(log_loss(y_test, predict))
pred = clf.predict_proba(X_test)
print(log_loss(y_test, pred))

#m = min(scores_test)  
#print(scores_test.index(min(scores_test)))