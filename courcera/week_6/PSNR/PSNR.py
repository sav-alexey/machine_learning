from sklearn.cluster import KMeans
from skimage.io import imread
from skimage import img_as_float
from skimage.measure import compare_psnr
import numpy as np
import pandas as pd

image = imread('parrots.jpg')
image = img_as_float(image)
matrix = np.reshape(image, (-1, image.shape[2]))

for j in range(9, 20):
    clst = KMeans(n_clusters=j, init='k-means++', random_state=241)
    clst.fit(matrix)
    pred = clst.fit_predict(matrix)
    
    mean = matrix.copy() 
    
    for i,value in enumerate(clst.cluster_centers_):        
        mean[clst.labels_ == i] = value
        
    X_median = matrix.copy()
    
    for i in range(clst.n_clusters):
        X_median[pred==i] = np.median(matrix[pred==i], axis=0)    
    print(compare_psnr(matrix, mean), compare_psnr(matrix, X_median))
#df = pd.DataFrame(image)
#df['cluster'] = clast.predict(df)