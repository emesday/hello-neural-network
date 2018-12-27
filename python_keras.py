from keras.layers import Input, Dense
from keras.models import Model
from python import load, random_state
from keras import regularizers

from numpy.random import seed
from tensorflow import set_random_seed
seed(random_state)
set_random_seed(random_state + 2)

if __name__ == '__main__':
    X_train, y_train = load('dataset/train.bin')
    X_test, y_test = load('dataset/test.bin')

    inputs = Input(shape=(X_train.shape[1],))
    layers = Dense(1, activation='sigmoid')(inputs)
    model = Model(inputs=inputs, outputs=layers)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, y_train, epochs=20)
    loss, accuracy = model.evaluate(X_test, y_test)
    print('accuracy: {}'.format(accuracy))
    # accuracy: 0.8771929835018358


