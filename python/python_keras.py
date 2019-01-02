from python import load_dataset, random_state, l2_penalty

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

import numpy as np
import tensorflow as tf

"""$ python python_keras.py
Using TensorFlow backend.
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 2)                 0
_________________________________________________________________
output (Dense)               (None, 1)                 3
=================================================================
Total params: 3
Trainable params: 3
Non-trainable params: 0
_________________________________________________________________
Epoch 1/10
1000/1000 [==============================] - 0s 85us/step - loss: 3.0713
Epoch 2/10
1000/1000 [==============================] - 0s 21us/step - loss: 0.8719
Epoch 3/10
1000/1000 [==============================] - 0s 24us/step - loss: 0.2511
Epoch 4/10
1000/1000 [==============================] - 0s 20us/step - loss: 0.0729
Epoch 5/10
1000/1000 [==============================] - 0s 25us/step - loss: 0.0233
Epoch 6/10
1000/1000 [==============================] - 0s 21us/step - loss: 0.0092
Epoch 7/10
1000/1000 [==============================] - 0s 21us/step - loss: 0.0051
Epoch 8/10
1000/1000 [==============================] - 0s 23us/step - loss: 0.0039
Epoch 9/10
1000/1000 [==============================] - 0s 21us/step - loss: 0.0036
Epoch 10/10
1000/1000 [==============================] - 0s 24us/step - loss: 0.0035
W: [ 0.79843241  1.60144722], b: [ 2.16585469]
"""

if __name__ == '__main__':
    x, y = load_dataset()
    # build model
    np.random.seed(random_state)
    tf.set_random_seed(random_state + 2)
    inputs = Input(shape=(x.shape[1],))
    layers = Dense(1, name='output',
                   kernel_regularizer=regularizers.l2(l2_penalty),
                   bias_regularizer=regularizers.l2(l2_penalty))(inputs) # no activation
    model = Model(inputs=inputs, outputs=layers)
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.summary()
    # training
    model.fit(x, y, epochs=10)
    # get weights
    W, b = model.get_layer('output').get_weights()
    print('W: %s, b: %s' % (W.flatten(), b))

