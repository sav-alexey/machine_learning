import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('titanic.csv', index_col='PassengerId')
data = data[['Survived', 'Pclass', 'Fare', 'Age', 'Sex']]
data = data.dropna()

X = data[['Pclass', 'Fare', 'Age', 'Sex']]
X['Sex'] = X['Sex'].map({'female': 1, 'male': 0})

y = data['Survived']


clf = DecisionTreeClassifier()
clf.fit(X, y)

X_test = [[3, 1.0, 1, 0]]

prediction = clf.predict(X_test)
#importances of features 
importances = clf.feature_importances_
print(importances, prediction)
