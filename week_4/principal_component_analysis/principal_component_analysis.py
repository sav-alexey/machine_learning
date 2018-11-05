import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

data = np.array(pd.read_csv("close_prices.csv"))
data_djia = pd.read_csv("djia_index.csv")
data = data[:,1:]
data_djia = data_djia.DJI
#for i in range(10, 4, -1):
#    princ = PCA(n_components= i)
#    princ.fit(data)
#    print(sum(princ.explained_variance_ratio_))

princ = PCA(n_components= 10)
princ.fit(data)
t = princ.transform(data)
X = [x[0] for x in t] #Значения первой компоненты
print(np.corrcoef(X, data_djia))