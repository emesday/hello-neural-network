from tensorflow import keras

# This downloads 4 files to ~/.keras/datasets/fashion-mnist
keras.datasets.fashion_mnist.load_data()

"""
$ ls -s -1 ~/.keras/datasets/fashion-mnist
total 61088
 8640 t10k-images-idx3-ubyte.gz
   16 t10k-labels-idx1-ubyte.gz
52368 train-images-idx3-ubyte.gz
   64 train-labels-idx1-ubyte.gz
"""
