import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data.astype('float32'), data.target.astype('int32'),
                                                    test_size=0.2, random_state=27,
                                                    stratify=data.target)

with open('train.bin', 'wb') as o:
    np.array(X_train.shape, dtype='int32').tofile(o)
    X_train.tofile(o)
    y_train.tofile(o)

with open('test.bin', 'wb') as o:
    np.array(X_test.shape, dtype='int32').tofile(o)
    X_test.tofile(o)
    y_test.tofile(o)
