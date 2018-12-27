import numpy as np


random_state=816

def load(infile):
    with open(infile, 'rb') as f:
        shape = np.fromfile(f, count=2, dtype='int32')
        X = np.fromfile(f, count=shape[0] * shape[1], dtype='float32').reshape(shape)
        y = np.fromfile(f, count=shape[0], dtype='int32')
    return X, y

