import pandas as p
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import math

data = p.read_csv('classification.csv')

data_score = p.read_csv('scores.csv')

true_score = data_score.true
sc1 = data_score.score_logreg
sc2 = data_score.score_svm
sc3 = data_score.score_knn
sc4 = data_score.score_tree

pred = data.pred
true = data.true
#pred = data[:,1:]
#true = data[:,0]

right = 107
false = 93

conf_matrix = confusion_matrix(data['true'], data['pred'], labels=[1, 0])

count = data.pred == data.true

a = []
for i in count:
    if i == False:
        a.append("1")

print(conf_matrix.T)

precision = 43/(34+43)
recall = 43/(43+59)

accuracy = accuracy_score(true, pred)
prec = precision_score(true, pred)

f = (precision*recall*2)/(precision+recall)
ac = (43+64)/200
roc1 = roc_auc_score(true_score, sc1)
roc2 = roc_auc_score(true_score, sc2)
roc3 = roc_auc_score(true_score, sc3)
roc4 = roc_auc_score(true_score, sc4)

#for score from sc1 to sc4
recall_index = []
pre_rec_curve1 = precision_recall_curve(true_score, sc4)[1]
for idx, i in enumerate(pre_rec_curve1):
    if i > 0.7 :
        recall_index.append(idx)

recall_array = []
for i in recall_index:
    recall_array.append(precision_recall_curve(true_score, sc4)[0][i])
    
    
#print(classification_report(true, pred))        
print(ac, precision, recall, f)
print(accuracy, prec)
print(roc1, roc2, roc3, roc4)
print(pre_rec_curve1)
print("max = ", max(recall_array))

