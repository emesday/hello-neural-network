import numpy as np

random_state=2019

def load_dataset():
    dataset = np.loadtxt('../dataset/dataset.txt', dtype=np.float32)
    x = dataset[:, 1:]
    y = dataset[:, 0]
    return x, y

if __name__ == '__main__':
    pass
