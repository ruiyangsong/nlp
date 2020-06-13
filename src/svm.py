#!/usr/bin/env python
import os, sys
import numpy as np
from utils import split_data, test_score, to_one_hot

os.environ['MKL_NUM_THREADS'] = '3'
os.environ['OPENBLAS_NUM_THREADS'] = '3'
np.set_printoptions(linewidth=np.inf)

def grid_search():
    mode_lst     = ['padding', 'sum']
    C_lst        = [1., 10., 100., 1000.] # Regularization parameter
    kernel_lst   = ['linear', 'poly', 'rbf', 'sigmoid']

    verbose = 0

    hyper_tag_lst = []
    acc_lst       = []
    for mode in mode_lst:
        for C in C_lst:
            for kernel in kernel_lst:
                print('\nmode_%s_C_%s_kernel_%s'%(mode,C,kernel))
                data_pth = '../data/mode_%s.npz' % mode
                x_train, y_train, x_test, y_test, x_val, y_val = _data(data_pth, split_val=True, verbose=verbose)

                svm_clf = SupportVectorMachine(C=C, kernel=kernel, verbose=verbose)
                svm_clf.fit(x=x_train, y=y_train)
                y_pred_val = svm_clf.predict(x_val, y_val, Onsave=False)
                acc, f1, mcc, recalls, precisions, f1s, mccs = test_score(y_real=y_val, y_pred=y_pred_val, classes=10)

                hyper_tag_lst.append('%s_%s_%s'%(mode, C, kernel))
                acc_lst.append(acc)
                print('val_acc:', acc)
                # sys.stdout.flush()# or "python -u“
    print('\nThe Best Hypers are: %s, Best val_acc is: %s' %(hyper_tag_lst[acc_lst.index(max(acc_lst))], max(acc_lst)))

def main():
    if len(sys.argv) < 4:
        print('Usage: %s %s %s %s'%(sys.argv[0], ['padding','sum'], 'C', 'kernel'))
        exit(0)

    MODE          = sys.argv[1]
    C             = float(sys.argv[2])
    KERNEL        = sys.argv[3]

    verbose = 1
    data_pth = '../data/mode_%s.npz' % MODE
    outdir = '../model/svm/mode_%s_C_%s_kernel_%s' % (MODE,C,KERNEL)
    os.makedirs(outdir, exist_ok=True)
    x_train, y_train, x_test, y_test = _data(data_pth, split_val=False, verbose=verbose)

    svm_clf = SupportVectorMachine(C=C,kernel=KERNEL,verbose=verbose)
    svm_clf.fit(x=x_train, y=y_train)
    y_pred = svm_clf.predict(x_test, y_test, modeldir=outdir, Onsave=True)
    acc, f1, mcc, recalls, precisions, f1s, mccs = test_score(y_real=y_test, y_pred=y_pred, classes=10)
    print('\nacc: %s'
          '\nf1: %s'
          '\nmcc: %s'
          '\nrecalls: %s'
          '\nprecisions: %s'
          '\nf1s: %s'
          '\nmccs: %s'%(acc, f1, mcc, recalls, precisions, f1s, mccs))
    print('\nThe Hypers are: mode_%s_C_%s_kernel_%s'%(MODE, C, KERNEL))

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

class SupportVectorMachine(object):
    def __init__(self, C=1.0, kernel='rbf',verbose=0):
        from sklearn.svm import SVC
        self.clf = SVC(decision_function_shape='ovr', C=C, kernel=kernel, verbose=verbose, probability=True)

    def fit(self, x, y):
        '''fit model by one vs. rest binary classification'''
        self.clf.fit(x, y)

    def predict(self, x_test, y_test, modeldir='../model/svm', Onsave=True):
        y_real = y_test
        y_pred = self.clf.predict(x_test)
        p_real = to_one_hot(y_test)
        p_pred = self.clf.predict_proba(x_test)
        if Onsave:
            try:
                np.savez('%s/test_rst.npz' % modeldir, y_real=y_real, y_pred=y_pred, p_real=p_real, p_pred=p_pred)
            except:
                print('save test_rst failed')
        return y_pred

if __name__ == "__main__":
    if len(sys.argv) == 1:
        grid_search()
    else:
        main()