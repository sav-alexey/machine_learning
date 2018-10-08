import tensorflow as tf
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = np.float32(iris.data[:100, [2,3]]).T
Y = np.float32(iris.target[:100])

Y = Y.reshape([100, 1])


tf.reset_default_graph() 
tf.set_random_seed(1)  
W1 = tf.get_variable("W1", [5, 2], initializer =  tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", [5, 1], initializer = tf.zeros_initializer())
W2 = tf.get_variable("W2", [1, 5], initializer =  tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", [1, 1], initializer = tf.zeros_initializer())
#    
Z1 = tf.add(tf.matmul(W1, X), b1)
A1 = tf.nn.relu(Z1)
Z2 = tf.add(tf.matmul(W2, A1), b2)
A2 = tf.sigmoid(Z2)

Y = tf.transpose(Y)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z2, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)

y = []
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10000):        
        costx = sess.run([optimizer, cost])[1]
        y.append(costx)


plt.ylabel('cost')
plt.xlabel('iterations')
plt.plot([i for i in range(10000)], y)