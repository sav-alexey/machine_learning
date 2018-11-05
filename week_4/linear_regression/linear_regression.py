import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

data_train = pd.read_csv('salary-train.csv')


for i in data_train.FullDescription:
    i.lower()
    re.sub('[^a-zA-Z0-9]', ' ', i)

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))


vect=TfidfVectorizer(min_df = 5)
X_train = vect.fit_transform(data_train.FullDescription)

X = hstack([X_train ,X_train_categ])




#X_train_categ = X_train_categ[['LocationNormalized', 'ContractTime']]

y_train = data_train.SalaryNormalized

clf = Ridge(alpha=1.0, random_state=241)
clf.fit(X, y_train)



data_test = pd.read_csv('salary-test-mini.csv')
for i in data_test.FullDescription:
    i.lower()
    re.sub('[^a-zA-Z0-9]', ' ', i)

data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = vect.transform(data_test.FullDescription)
X_test = hstack([X_test ,X_test_categ])


prediction = clf.predict(X_test)
print(prediction)