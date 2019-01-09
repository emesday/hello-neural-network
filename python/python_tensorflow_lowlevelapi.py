import tensorflow as tf
from tensorflow import keras

# Load the Fashion-MNIST dataset.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape(-1, 784)
test_images = test_images.reshape(-1, 784)
train_labels = tf.one_hot(train_labels, 10)
test_labels = tf.one_hot(test_labels, 10)

# Scale to a range of 0 to 1
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# tf.data
training_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
training_dataset = training_dataset.batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(32)

# reinitializable iterator for feeding training/test dataset
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
x, y = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
test_init_op = iterator.make_initializer(test_dataset)

# Setup the layers
W1 = tf.get_variable("W1", [784, 128], initializer=tf.initializers.glorot_uniform)
b1 = tf.get_variable("b1", [128], initializer=tf.zeros_initializer())
W2 = tf.get_variable("W2", [128, 10], initializer=tf.initializers.glorot_uniform)
b2 = tf.get_variable("b2", [10], initializer=tf.zeros_initializer())

z1 = tf.add(tf.matmul(x, W1), b1)
a1 = tf.nn.relu(z1)
z2 = tf.add(tf.matmul(a1, W2), b2)
a2 = tf.nn.softmax(z2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z2, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(loss)
meanloss = tf.metrics.mean(loss)
accuracy = tf.metrics.accuracy(tf.argmax(y, 1), tf.argmax(a2, 1))

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())  # for meanloss and accuracy
    sess.run(tf.global_variables_initializer())
    for epoch in range(5):
        sess.run(training_init_op)
        print('Epoch %d/5' % (epoch + 1))
        n = 0
        while True:
            n += 1
            try:
                sess.run([optimizer, meanloss, accuracy])
            except tf.errors.OutOfRangeError:
                break
        print(n, n * 32)
        print('loss: %.4f - acc: %.4f' % tuple(sess.run([meanloss[0], accuracy[0]])))

    sess.run(tf.local_variables_initializer())  # for accuracy
    sess.run(test_init_op)  # reinitialize for feeding test_dataset
    while True:
        try:
            sess.run(accuracy)
        except tf.errors.OutOfRangeError:
            break
    print('Test accuracy:', sess.run(accuracy[0]))

