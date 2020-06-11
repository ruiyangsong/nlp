import os, sys
import numpy as np
from utils import split_data, test_score

def main():
    if len(sys.argv) == 1:
        print('Usage: %s %s'%(sys.argv[0], ['stack','padding','sum']))
        exit(0)
    data_pth = '../data/mode_%s.npz' % sys.argv[1]
    outdir   = '../model/LR/mode_%s' % sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    x_train, y_train, x_test, y_test, x_val, y_val = _data(data_pth)

    MAX_ITER = 10000
    LEARNING_RATE = 0.1
    LR = LogisticRegression(max_iter=MAX_ITER, lr=LEARNING_RATE)
    thetas = LR.fit(x=x_train, y=y_train, reduce_lr=True, verbose=1)
    y_pred_val = LR.predict(thetas, x_val, y_val, Onsave=False)
    acc, recalls, precisions, f1s, mccs = test_score(y_real=y_val, y_pred=y_pred_val, classes=10)
    print('\nval_acc:',acc)
    y_pred = LR.predict(thetas, x_test, y_test, modeldir=outdir, Onsave=True)

def _data(data_pth):
    data = np.load(data_pth, allow_pickle=True)
    x, y = data['x'], data['y']
    x_train, y_train, x_test, y_test = split_data(x, y)
    x_train, y_train, x_val, y_val = split_data(x_train, y_train)

    print('\nx_train shape: %s'
          '\ny_train shape: %s'
          '\nx_test shape: %s'
          '\ny_test shape: %s'
          % (x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    return x_train, y_train, x_test, y_test, x_val, y_val

class LogisticRegression(object):
    def __init__(self, max_iter=5000, lr=0.1):
        self.max_iteration = max_iter
        self.learning_rate = lr

    def _sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def _cost(self,theta, x, y):
        h = self._sigmoid(x @ theta)
        m = len(y)
        cost = 1 / m * np.sum(
            -y * np.log(h) - (1 - y) * np.log(1 - h)
        )
        grad = 1 / m * ((y - h) @ x)
        return cost, grad

    def _reduceLR(self, cost, min_delta=1e-4, patience=100, factor=0.1, verbose=1):
        if len(cost) > patience and max(cost[-patience-1:-1]) - cost[-1] < min_delta:
            self.learning_rate = self.learning_rate * factor
            if verbose:
                print('\n--reduce learning_rate to %.8f'%self.learning_rate)

    def fit(self, x, y, reduce_lr=True, verbose=1):
        '''fit model by one vs. rest binary classification'''
        x = np.insert(x, 0, 1, axis=1) # integrate bias to theta
        classes = np.unique(y)
        thetas  = []
        for c in classes:
            print('\n@Fitting class %s...'%c)
            binary_y = np.where(y == c, 1, 0) # one-hot label
            theta    = np.zeros(x.shape[1]) # init theta to zeros
            cost     = []
            for epoch in range(self.max_iteration):
                _cost, _grad = self._cost(theta, x, binary_y)
                cost.append(_cost)
                theta += self.learning_rate * _grad
                if reduce_lr:
                    self._reduceLR(cost)
                    if self.learning_rate < 1e-5:
                        print('\n --lr is too small, early stopping was called.')
                        break
                if verbose and epoch % 1000 == 0:
                    print('\nepoch --> %s'
                          '\ncost --> %s'
                          '\ngrad[:2] --> %s'
                          '\ntheta[:2] --> %s'%(epoch, cost[-1], _grad[:2], theta[:2]))
            thetas.append(theta)
        return thetas

    def predict(self, thetas, x_test, y_test, modeldir='../model/LR', Onsave=True):
        x = np.insert(x_test, 0, 1, axis=1)
        y_pred = [np.argmax(
            [self._sigmoid(xi @ theta) for theta in thetas]
        ) for xi in x]
        y_real = y_test
        p_pred = [[self._sigmoid(xi @ theta) for theta in thetas] for xi in x]
        #
        # save thetas, p_pred of each bi-classifiers
        #
        if Onsave:
            try:
                np.savez('%s/test_rst.npz' % modeldir, y_real=y_real, y_pred=y_pred, p_pred=p_pred)
                np.savez('%s/thetas.npz' % modeldir, thetas=thetas)
            except:
                print('save thetas and test_rst failed')
        return y_pred

if __name__ == '__main__':
    main()
