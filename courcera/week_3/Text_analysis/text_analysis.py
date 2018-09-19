import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

target=newsgroups.target
vect=TfidfVectorizer(use_idf=True)
idf=vect.fit_transform(newsgroups.data)

kf = KFold(n_splits=5, shuffle = True, random_state= 42)


counter = range(-5, 5)
a = []
for i in counter:
    a.append(10**i)


clf = SVC(C = 1, kernel='linear', random_state=241)
clf.fit(idf, target)

b = np.argsort(np.abs(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]


final = []
for i in b:
    a = vect.get_feature_names()[i]
    final.append(a)
    
final.sort()
print(final)