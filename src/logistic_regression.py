#!/usr/bin/env python
import os
import sys
import numpy as np
from utils import split_data, test_score

os.environ['MKL_NUM_THREADS'] = '3'
os.environ['OPENBLAS_NUM_THREADS'] = '3'
np.set_printoptions(linewidth=np.inf)

def grid_search():
    mode_lst     = ['padding', 'sum']
    max_iter_lst = [100, 500, 1000, 5000, 10000]
    learning_rate_lst = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    hyper_tag_lst = []
    acc_lst       = []
    for mode in mode_lst:
        for max_iter in max_iter_lst:
            for learning_rate in learning_rate_lst:
                print('\nmode_%s_iter_%s_lr_%s'%(mode,max_iter,learning_rate))
                data_pth = '../data/mode_%s.npz' % mode
                x_train, y_train, x_test, y_test, x_val, y_val = _data(data_pth, split_val=True)

                MAX_ITER = max_iter
                LEARNING_RATE = learning_rate
                verbose = 0

                LR = LogisticRegression(max_iter=MAX_ITER, lr=LEARNING_RATE)
                thetas = LR.fit(x=x_train, y=y_train, reduce_lr=True, verbose=verbose)
                y_pred_val = LR.predict(thetas, x_val, y_val, Onsave=False)
                acc, recalls, precisions, f1s, mccs = test_score(y_real=y_val, y_pred=y_pred_val, classes=10)

                hyper_tag_lst.append('%s_%s_%s'%(mode, max_iter, learning_rate))
                acc_lst.append(acc)
                print('val_acc:', acc)
                # sys.stdout.flush()# or "python -u“
    print('\nThe Best Hypers are: %s, Best val_acc is: %s' %(hyper_tag_lst[acc_lst.index(max(acc_lst))], max(acc_lst)))

def main():
    if len(sys.argv) == 1:
        print('Usage: %s %s %s %s'%(sys.argv[0], ['padding','sum'], 'max_iter', 'learning_rate'))
        exit(0)

    data_pth      = '../data/mode_%s.npz' % sys.argv[1]
    outdir        = '../model/LR/mode_%s' % sys.argv[1]
    MAX_ITER      = int(sys.argv[2])
    LEARNING_RATE = float(sys.argv[3])
    verbose = 1
    os.makedirs(outdir, exist_ok=True)
    x_train, y_train, x_test, y_test = _data(data_pth, split_val=False, verbose=1)

    LR = LogisticRegression(max_iter=MAX_ITER, lr=LEARNING_RATE)
    thetas = LR.fit(x=x_train, y=y_train, reduce_lr=True, verbose=verbose)
    y_pred = LR.predict(thetas, x_test, y_test, modeldir=outdir, Onsave=True)
    acc, recalls, precisions, f1s, mccs = test_score(y_real=y_test, y_pred=y_pred, classes=10)
    print('\nacc: %s'
          '\nrecalls: %s'
          '\nprecisions: %s'
          '\nf1s: %s'
          '\nmccs: %s'%(acc, recalls, precisions, f1s, mccs))
    print('\nThe Hypers are: mode_%s_iter_%s_lr_%s'%(sys.argv[1],sys.argv[2],sys.argv[3]))

def _data(data_pth, split_val=True, verbose=0):
    data = np.load(data_pth, allow_pickle=True)
    x, y = data['x'], data['y']
    x_train, y_train, x_test, y_test = split_data(x, y)
    if split_val:
        x_train, y_train, x_val, y_val = split_data(x_train, y_train)
        if verbose:
            print('\nx_train shape: %s'
                  '\ny_train shape: %s'
                  '\nx_test shape: %s'
                  '\ny_test shape: %s'
                  '\nx_val shape: %s'
                  '\ny_val shape: %s'
                  % (x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape))
        return x_train, y_train, x_test, y_test, x_val, y_val
    else:
        if verbose:
            print('\nx_train shape: %s'
                  '\ny_train shape: %s'
                  '\nx_test shape: %s'
                  '\ny_test shape: %s'
                  % (x_train.shape, y_train.shape, x_test.shape, y_test.shape))
        return x_train, y_train, x_test, y_test

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
            if verbose:
                print('\n@Fitting class %s...'%c)
            binary_y = np.where(y == c, 1, 0) # one-hot label
            theta    = np.zeros(x.shape[1]) # init theta to zeros
            cost     = []
            for epoch in range(self.max_iteration):
                _cost, _grad = self._cost(theta, x, binary_y)
                cost.append(_cost)
                theta += self.learning_rate * _grad
                if reduce_lr:
                    self._reduceLR(cost, verbose=verbose)
                    if self.learning_rate < 1e-5 and verbose:
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
    if len(sys.argv) == 1:
        grid_search()
    else:
        main()
