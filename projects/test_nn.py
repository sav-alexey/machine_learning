import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

class Perceptron(object):    
    def __init__(self, eta=0.2, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

mylist1 = [[1,6], [1,7], [2,6], [3, 8]]
X = np.array([[6], [7], [6], [8], [5] , [2], [3], [1], [0], [1]])
y = np.array([0, 0, 0, 0, 0 , 1, 1, 1, 1, 1])
xlist = list(range(10))
#xlist = list(range(10))
#print(xlist, ylist)
#plt.scatter(xlist, ylist)

per = Perceptron()
per.fit(X, y)

clf = KNeighborsClassifier()
clf.fit(X, y)


print(per.predict(1))
print(clf.predict(8))
print(per.w_)

def func(x, a):
    y = []
    for i in x:
        y.append(a*i)
    return y
    
plt.scatter(X, xlist)
plt.plot(func(xlist, 0.1), xlist)