import pandas as p
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import math

data = p.read_csv('classification.csv')

pred = data.pred
true = data.true
#pred = data[:,1:]
#true = data[:,0]

right = 107
false = 93

conf_matrix = confusion_matrix(data['true'], data['pred'], labels=[1, 0])

count  = data.pred == data.true

a = []
for i in count:
    if i == False:
        a.append("1")

precision = 43/(43+64)
recall = 43/(43+59)

f = (precision*recall*2)/(precision+recall)

        
print(f, precision, recall)