#!/usr/bin/env python
import os, sys, time
import numpy as np
from keras import models, layers, optimizers, callbacks, regularizers, initializers
from utils import split_data, calc_class_weights, to_one_hot, config_tf, net_saver, net_predictor, test_score

os.environ['MKL_NUM_THREADS'] = '3'
os.environ['OPENBLAS_NUM_THREADS'] = '3'
np.set_printoptions(linewidth=np.inf)

def grid_search():
    mode_lst = ['padding', 'sum']
    epochs_lst = [200] #cuz early stopping exists
    learning_rate_lst = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    verbose = 0

    hyper_tag_lst = []
    acc_lst = []
    for mode in mode_lst:
        for epochs in epochs_lst:
            for learning_rate in learning_rate_lst:
                data_pth   = '../data/mode_%s.npz' %mode
                x_train, y_train, x_test, y_test, x_val, y_val, class_weights_dict = _data(data_pth,split_val=True,verbose=verbose)

                config_tf(user_mem=2500, cuda_rate=0.2)
                model, history_dict = TrainConv1D(x_train, y_train, x_val, y_val, class_weights_dict, filepth=None, epochs=epochs, lr=learning_rate,verbose=verbose)
                acc = history_dict['val_acc'][-1]
                early_epochs = len(history_dict['val_acc'])
                hyper_tag_lst.append('%s_%s_%s' % (mode, early_epochs, learning_rate))
                print('\nmode_%s_epochs_%s_lr_%s' % (mode, epochs, learning_rate))
                acc_lst.append(acc)
                print('val_acc:', acc)
                # sys.stdout.flush()# or "python -u“
    print('\nThe Best Hypers are: %s, Best val_acc is: %s' % (hyper_tag_lst[acc_lst.index(max(acc_lst))], max(acc_lst)))

def main():
    if len(sys.argv) < 4:
        print('Usage: %s %s %s %s'%(sys.argv[0], ['padding','sum'], 'epochs', 'learning_rate'))
        exit(0)
    MODE          = sys.argv[1]
    EPOCHS        = int(sys.argv[2])
    LEARNING_RATE = float(sys.argv[3])

    data_pth = '../data/mode_%s.npz' %MODE
    model_name = '%s_mode%s_%s'%(sys.argv[0].split('/')[-1][:-3], MODE, time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime()))
    modeldir   = '../model/Conv1D/%s' % model_name
    os.makedirs(modeldir, exist_ok=True)

    #
    # load data
    #
    x_train, y_train, x_test, y_test, class_weights_dict = _data(data_pth,split_val=False,verbose=1)

    #
    # train and save
    #
    config_tf(user_mem=2500, cuda_rate=0.2)
    model, history_dict = TrainConv1D(x_train, y_train, class_weights_dict=class_weights_dict, epochs=EPOCHS, lr=LEARNING_RATE)
    net_saver(model, modeldir, history_dict)

    #
    # test
    #
    y_test, p_pred, y_real, y_pred = net_predictor(modeldir, x_test, y_test, Onsave=True)
    acc, recalls, precisions, f1s, mccs = test_score(y_real=y_real, y_pred=y_pred, classes=10)
    print('\nacc: %s'
          '\nrecalls: %s'
          '\nprecisions: %s'
          '\nf1s: %s'
          '\nmccs: %s' % (acc, recalls, precisions, f1s, mccs))
    print('\nThe Hypers are: mode_%s_epochs_%s_lr_%s' % (MODE, EPOCHS, LEARNING_RATE))

def _data(data_pth, split_val=True, verbose=1):
    data = np.load(data_pth, allow_pickle=True)
    x, y = data['x'], data['y']
    x = x[:,:,np.newaxis]
    x_train, y_train, x_test, y_test = split_data(x, y)

    class_weights_dict = calc_class_weights(y_train)

    if split_val:
        x_train, y_train, x_val, y_val = split_data(x_train, y_train)
        y_train = to_one_hot(y_train, dimension=10)
        y_test = to_one_hot(y_test, dimension=10)
        y_val = to_one_hot(y_val, dimension=10)
        if verbose:
            print('\nx_train shape: %s'
                  '\ny_train shape: %s'
                  '\nx_test shape: %s'
                  '\ny_test shape: %s'
                  '\nx_val shape: %s'
                  '\ny_val shape: %s'
                  % (x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape))
        return x_train, y_train, x_test, y_test, x_val, y_val, class_weights_dict
    else:
        y_train = to_one_hot(y_train, dimension=10)
        y_test = to_one_hot(y_test, dimension=10)
        if verbose:
            print('\nx_train shape: %s'
                  '\ny_train shape: %s'
                  '\nx_test shape: %s'
                  '\ny_test shape: %s'
                  % (x_train.shape, y_train.shape, x_test.shape, y_test.shape))
        return x_train, y_train, x_test, y_test, class_weights_dict

def TrainConv1D(x_train, y_train, x_val=None, y_val=None, class_weights_dict=None, filepth=None, epochs=200, lr=1e-2, verbose=1):
    summary    = False
    batch_size = 128
    optimizer  = 'adam'
    activator  = 'relu'

    pool_size     = 2
    init_Conv1D   = initializers.lecun_uniform()
    init_Dense    = initializers.he_normal()
    padding_style = 'same'
    drop_rate     = 0.025
    l2_coeff      = 1e-3
    loss_type     = 'categorical_crossentropy'
    metrics       = ('acc',)

    loop_conv_num = 4 # 100 -> 50 -> 25 -> 13 -> 7 -> 4
    dense_num     = 128
    dropout_dense = 0.25

    if lr > 0:
        if optimizer == 'adam':
            chosed_optimizer = optimizers.Adam(lr=lr)
        elif optimizer == 'sgd':
            chosed_optimizer = optimizers.SGD(lr=lr)
        elif optimizer == 'rmsprop':
            chosed_optimizer = optimizers.RMSprop(lr=lr)

    if x_val is None or y_val is None:
        val_data = None
        my_callbacks = None
    else:
        val_data = (x_val, y_val)
        my_callbacks = [
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=5,
                verbose=verbose,
            ),
            callbacks.EarlyStopping(
                monitor='val_acc',
                min_delta=1e-4,
                patience=20,
                mode='max',
                verbose=verbose,
            ),
        ]
        if filepth is not None:
            my_callbacks+=[callbacks.ModelCheckpoint(filepath=filepth,
                                                     monitor='val_acc',
                                                     mode='max',
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     verbose=verbose,)]
    #
    # build model
    #
    network = models.Sequential()
    network.add(layers.SeparableConv1D(filters=16,
                                       kernel_size=5,
                                       activation=activator,
                                       padding=padding_style,
                                       depthwise_initializer=init_Conv1D,
                                       pointwise_initializer=init_Conv1D,
                                       depthwise_regularizer=regularizers.l2(l2_coeff),
                                       pointwise_regularizer=regularizers.l1(l2_coeff),
                                       input_shape=(x_train.shape[1:])))
    network.add(layers.BatchNormalization(axis=-1))
    network.add(layers.Dropout(drop_rate))
    network.add(layers.MaxPooling1D(pool_size=pool_size, padding=padding_style))

    for _ in range(loop_conv_num):
        network.add(layers.SeparableConv1D(filters=32,
                                           kernel_size=5,
                                           activation=activator,
                                           padding=padding_style,
                                           depthwise_initializer=init_Conv1D,
                                           pointwise_initializer=init_Conv1D,
                                           depthwise_regularizer=regularizers.l2(l2_coeff),
                                           pointwise_regularizer=regularizers.l1(l2_coeff),))
        network.add(layers.BatchNormalization(axis=-1))
        network.add(layers.Dropout(drop_rate))
        network.add(layers.MaxPooling1D(pool_size=pool_size, padding=padding_style))

    network.add(layers.SeparableConv1D(filters=64,
                                       kernel_size=3,
                                       activation=activator,
                                       padding=padding_style,
                                       depthwise_initializer=init_Conv1D,
                                       pointwise_initializer=init_Conv1D,
                                       depthwise_regularizer=regularizers.l2(l2_coeff),
                                       pointwise_regularizer=regularizers.l1(l2_coeff),))
    network.add(layers.BatchNormalization(axis=-1))
    network.add(layers.Dropout(drop_rate))
    network.add(layers.MaxPooling1D(pool_size=pool_size, padding=padding_style))

    network.add(layers.Flatten())
    network.add(layers.Dense(units=dense_num, kernel_initializer=init_Dense, activation=activator))
    network.add(layers.Dropout(dropout_dense))
    network.add(layers.Dense(units=10, kernel_initializer=init_Dense, activation='softmax'))

    if summary:
        print(network.summary())

    network.compile(optimizer=chosed_optimizer,
                    loss=loss_type,
                    metrics=list(metrics))
    result = network.fit(x=x_train,
                         y=y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=verbose,
                         callbacks=my_callbacks,
                         validation_data=val_data,
                         shuffle=True,
                         class_weight=class_weights_dict
                         )
    return network, result.history

if __name__ == '__main__':
    if len(sys.argv) == 1:
        grid_search()
    else:
        main()