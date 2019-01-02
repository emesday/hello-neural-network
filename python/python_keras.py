from python import load_dataset, random_state

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

from numpy.random import seed
from tensorflow import set_random_seed
seed(random_state)
set_random_seed(random_state + 2)

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
None
Epoch 1/10
1000/1000 [==============================] - 0s 84us/step - loss: 3.0712
Epoch 2/10
1000/1000 [==============================] - 0s 27us/step - loss: 0.8717
Epoch 3/10
1000/1000 [==============================] - 0s 24us/step - loss: 0.2507
Epoch 4/10
1000/1000 [==============================] - 0s 26us/step - loss: 0.0725
Epoch 5/10
1000/1000 [==============================] - 0s 21us/step - loss: 0.0229
Epoch 6/10
1000/1000 [==============================] - 0s 23us/step - loss: 0.0088
Epoch 7/10
1000/1000 [==============================] - 0s 21us/step - loss: 0.0047
Epoch 8/10
1000/1000 [==============================] - 0s 20us/step - loss: 0.0035
Epoch 9/10
1000/1000 [==============================] - 0s 24us/step - loss: 0.0031
Epoch 10/10
1000/1000 [==============================] - 0s 20us/step - loss: 0.0030
W: [ 0.79843926  1.60144794], b: [ 2.16606855]
"""

if __name__ == '__main__':
    x, y = load_dataset()

    inputs = Input(shape=(x.shape[1],))
    layers = Dense(1, kernel_regularizer=regularizers.l2(0.0001), name='output')(inputs) # no activation
    model = Model(inputs=inputs, outputs=layers)
    model.compile(loss='mean_squared_error', optimizer='sgd')
    print(model.summary())

    model.fit(x, y, epochs=10)

    W, b = model.get_layer('output').get_weights()
    print('W: %s, b: %s' % (W.flatten(), b))

