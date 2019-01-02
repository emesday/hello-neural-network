from python import load_dataset, random_state, l2_penalty

import numpy as np
import tensorflow as tf

"""$ python python_tensorflow.py
loss: 5.20069
loss: 3.37459
loss: 2.1902
loss: 1.42194
loss: 0.923573
loss: 0.600251
loss: 0.390476
loss: 0.25436
loss: 0.166032
loss: 0.108711
loss: 0.0715082
loss: 0.0473612
loss: 0.0316871
loss: 0.0215121
loss: 0.0149064
loss: 0.0106176
loss: 0.00783287
loss: 0.00602466
loss: 0.00485041
loss: 0.00408782
loss: 0.00359256
loss: 0.00327087
loss: 0.00306192
loss: 0.00292619
loss: 0.00283801
loss: 0.00278073
loss: 0.00274351
loss: 0.00271933
loss: 0.00270361
loss: 0.0026934
W: [ 0.79794234  1.60054266], b: [ 2.16693139]
"""

if __name__ == '__main__':
    x, y = load_dataset()
    # build model
    np.random.seed(random_state)
    tf.set_random_seed(random_state + 2)
    W = tf.Variable(tf.random_uniform([1, 2], -0.5, 0.5))
    b = tf.Variable(tf.zeros([1]))
    pred = tf.matmul(W, x.T) + b
    loss = tf.reduce_mean(tf.square(pred - y))
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    # training
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(30):
            sess.run(train)
            print("loss: %s" % sess.run(loss))
        # get weights
        print('W: %s, b: %s' % (sess.run(W).flatten(), sess.run(b)))

