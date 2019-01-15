import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the Fashion-MNIST dataset.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Scale to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# label 0 is the positive examples of the new label 0
# label 1 is the negative examples of the new label 0
# label 2 is the positive examples of the new label 1
# label 3 is the negative examples of the new label 1
# ...
# label 8 is the positive examples of the new label 4
# label 9 is the negative examples of the new label 5

label_map = np.empty((10, 5))
label_map[:] = -1

for i in range(5):
    label_map[i * 2][i] = 1
    label_map[i * 2 + 1][i] = 0

train_labels = label_map[train_labels]
test_labels = label_map[test_labels]

## Setup the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(5, activation=tf.nn.sigmoid)
])


def sparse_sigmoid_cross_entory(
        multi_class_labels, logits, weights=1.0, label_smoothing=0, scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    if multi_class_labels is None:
        raise ValueError("multi_class_labels must not be None.")
    if logits is None:
        raise ValueError("logits must not be None.")
    with tf.name_scope(scope, "sparse_sigmoid_cross_entropy_loss",
                       (logits, multi_class_labels, weights)) as scope:
        logits = tf.convert_to_tensor(logits)
        multi_class_labels = tf.cast(multi_class_labels, logits.dtype)
        logits.get_shape().assert_is_compatible_with(multi_class_labels.get_shape())
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=multi_class_labels,
                                                         logits=logits,
                                                         name="xentropy")
        zeros = tf.zeros_like(logits)
        losses = tf.where(multi_class_labels < 0, zeros, losses)
        return tf.losses.compute_weighted_loss(
            losses, weights, scope, loss_collection, reduction=reduction)


def accuracy(y_true, y_pred, threshold=0.5):
    threshold = tf.cast(threshold, y_pred.dtype)
    y_pred = tf.cast(y_pred > threshold, y_pred.dtype)
    indices = tf.where(y_true >= 0)
    return tf.reduce_mean(tf.cast(tf.equal(tf.gather_nd(y_true, indices), tf.gather_nd(y_pred, indices)), tf.float32),
                          axis=-1)


# Compile the model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=sparse_sigmoid_cross_entory,
              metrics=[accuracy])

model.summary()

# Train the model
model.fit(train_images, train_labels, epochs=2, batch_size=32)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

t = 0
n = 0

for label, pred in zip(test_labels, model.predict(test_images, 1024)):
    y_true = label[label >= 0].astype(np.float32)
    y_pred = (pred[label >= 0] > 0.5).astype(np.float32)
    t += y_true == y_pred
    n += 1

print("isclose: %s" % np.isclose(test_acc, t / n))


