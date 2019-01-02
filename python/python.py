import numpy as np

random_state = 2019
learning_rate = 0.1
l2_penalty = 0.0001

def load_dataset():
    dataset = np.loadtxt('../dataset/dataset.txt', dtype=np.float32)
    x = dataset[:, 1:]
    y = dataset[:, 0]
    return x, y

"""$ python python.py
loss: [ 0.06480751]
loss: [ 0.00320448]
W: [ 0.79447335  1.600678  ], b: [ 2.14965207]
"""

if __name__ == '__main__':
    x, y = load_dataset()

    # build model
    np.random.seed(random_state)
    W = np.random.normal(-0.5, 0.5, 2)
    b = np.zeros([1])
    get_score = lambda x: np.dot(x, W) + b
    # training
    for epoch in range(2):
        loss = 0
        for example, target in zip(x, y):
            score = get_score(example)
            grad = score - target
            loss += grad * grad
            W -= learning_rate * grad * example + l2_penalty * W.prod()
            b -= learning_rate * grad + l2_penalty * b
        print('loss: %s' % (loss / x.shape[0]))
    # get weights
    print('W: %s, b: %s' % (W, b))
