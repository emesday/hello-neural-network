from python import load, random_state
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    X_train, y_train = load('dataset/train.bin')
    X_test, y_test = load('dataset/test.bin')
    
    lr = LogisticRegression(random_state=random_state)
    lr.fit(X_train, y_train)
    
    accuracy = lr.score(X_test, y_test)
    
    print('accuracy: {}'.format(accuracy))
    # accuracy: 0.956140350877193

