import pandas as p
import numpy as np
import sklearn.metrics
import math

data = np.array(p.read_csv('data-logistic.csv', header = None))
X = data[:,1:]
y = data[:,0]

def sigmoid(x):
    return 1.0/(1 + math.exp(x))

def distance(a, b):
    return np.sqrt(np.square(a[0]-b[0]) + (a[1]-b[1]))

def log_regression(X, y, k, w, C, epsilon, max_iter):
    w1, w2 = w
    predictions = []
    for i in range(max_iter):
        w1_new = w1 + k * np.mean(y*X[:,0] * (1-(1./(1+np.exp(-y*(w1*X[:,0] + w2*X[:,1])))))) - k*C*w1
        w2_new = w2 + k * np.mean(y*X[:,1] * (1-(1./(1+np.exp(-y*(w1*X[:,0] + w2*X[:,1])))))) - k*C*w2
        if distance((w1_new, w2_new), (w1, w2)) < epsilon:
            break
        w1, w2 = w1_new, w2_new
    
    for i in range(len(X)):
        t1 = -w1*X[i, 0] - w2*X[i, 1]
        s = sigmoid(t1)
        predictions.append(s)
    return predictions

p0 = log_regression(X, y, 0.1, [0.0, 0.0], 0, 0.00001, 10000)
p1 = log_regression(X, y, 0.1, [0.0, 0.0], 10, 0.00001, 10000)    
        
print(sklearn.metrics.roc_auc_score(y, p0)) 
print(sklearn.metrics.roc_auc_score(y, p1))       

