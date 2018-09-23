import math
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def neur(X, y):
    w = 0.1
    
#    w_ideal = 1
    for i in range(10):
        error = []
        for x, w_ideal in zip(X, y):        
            h = w*x
            out = w_ideal - h
            error.append(out**2)            
            gr = h*out
            delta = gr
            d_w = 0.4*gr
            w += d_w
        print(sum(error)/len(error))            
    return w



X = [9, 8, 7, 1, 2, 3]
y = [1, 1, 1, 0, 0, 0]

w = neur(X, y)

def pred(w, X):
    if X*w > 0.0:
        prediction = 1
    else:
        prediction = 0
    return prediction
            
print(pred(w, 1))    

   
#neur(3)  
#
#def func(xlist, h):
#    y = []
#    for i in xlist:
#        y.append(h*i)
#    return y
#
#xlist = list(range(10)) 
#plt.plot(func(xlist, neur(1)), xlist)