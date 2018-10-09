import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.array(pd.read_csv("wine.data.csv"))
Y_data = data[:,0]
X_data = data[:,1:]
#Y_data = Y_data.reshape([177, 1])


tf.reset_default_graph() 
tf.set_random_seed(1)  
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", [5, 13], initializer =  tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", [5, 1], initializer = tf.zeros_initializer())
W2 = tf.get_variable("W2", [3, 5], initializer =  tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", [1, 1], initializer = tf.zeros_initializer())

Z1 = tf.add(tf.matmul(W1, X), b1)
A1 = tf.nn.relu(Z1)
Z2 = tf.add(tf.matmul(W2, A1), b2)
A2 = tf.nn.softmax(Z2)

Y = tf.transpose(Y)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z2, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
acc, acc_op = tf.metrics.accuracy(labels = X, predictions = Y)

init = tf.global_variables_initializer()
local = tf.local_variables_initializer()

y = []
with tf.Session() as sess:
    sess.run(init)
    sess.run(local)
#    y_cap = sess.run(A2, feed_dict={X:X_data.T})
#    print(y_cap)
    for epoch in range(1000):        
        costx = sess.run([optimizer, cost], feed_dict={X:X_data.T, Y:Y_data})[1]
        y.append(costx)
    y_cap = sess.run(A2, feed_dict={X:X_data.T})
    print(y_cap)
        
plt.ylabel('cost')
plt.xlabel('iterations')
plt.plot([i for i in range(1000)], y)