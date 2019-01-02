from python import load_dataset, random_state
from sklearn.linear_model import SGDRegressor

"""$ python python_sklearn.py
-- Epoch 1
Norm: 1.59, NNZs: 2, Bias: 1.941867, T: 1000, Avg. loss: 0.660085
Total training time: 0.00 seconds.
-- Epoch 2
Norm: 1.75, NNZs: 2, Bias: 2.123407, T: 2000, Avg. loss: 0.014382
Total training time: 0.00 seconds.
-- Epoch 3
Norm: 1.78, NNZs: 2, Bias: 2.158143, T: 3000, Avg. loss: 0.001975
Total training time: 0.00 seconds.
-- Epoch 4
Norm: 1.79, NNZs: 2, Bias: 2.166933, T: 4000, Avg. loss: 0.001385
Total training time: 0.00 seconds.
Convergence after 4 epochs took 0.00 seconds
W: [ 0.79767967  1.60081617], b: [ 2.16693286]
"""

if __name__ == '__main__':
    x, y = load_dataset()
    model = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, random_state=random_state, max_iter=1000, tol=1e-3, verbose=1)
    model.fit(x, y)
    print('W: %s, b: %s' % (model.coef_, model.intercept_))


