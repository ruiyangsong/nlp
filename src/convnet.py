#!/usr/bin/env python
import os, sys, time
import numpy as np
from keras import models, layers, optimizers, callbacks, regularizers, initializers
from utils import split_data, calc_class_weights, to_one_hot, config_tf, net_saver, net_predictor

def main():
    if len(sys.argv) == 1:
        print('Usage: %s %s'%(sys.argv[0], ['stack','padding','sum']))
        exit(0)

    data_pth   = '../data/mode_%s.npz' % sys.argv[1]
    model_name = '%s_mode%s_%s'%(sys.argv[0].split('/')[-1][:-3], sys.argv[1], time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime()))
    modeldir   = '../model/Conv1D/%s' % model_name
    filepth    = '%s/weights-best.h5' % modeldir
    os.makedirs(modeldir, exist_ok=True)

    #
    # load data
    #
    x_train, y_train, x_test, y_test, x_val, y_val, class_weights_dict = _data(data_pth)

    #
    # train and save
    #
    config_tf(user_mem=2500, cuda_rate=0.2)
    model, history_dict = TrainConv1D(x_train, y_train, x_val, y_val, class_weights_dict, filepth)
    net_saver(model, modeldir, history_dict)

    #
    # test
    #
    y_test, p_pred, y_real, y_pred = net_predictor(modeldir, x_test, y_test)

def _data(data_pth):
    data = np.load(data_pth, allow_pickle=True)
    x, y = data['x'], data['y']
    x_train, y_train, x_test, y_test = split_data(x, y)
    x_train, y_train, x_val, y_val = split_data(x_train, y_train)

    x_train = x_train[:,:,np.newaxis]
    x_test  = x_test[:,:,np.newaxis]
    x_val   = x_val[:,:,np.newaxis]

    class_weights_dict = calc_class_weights(y_train)

    y_train = to_one_hot(y_train, dimension=10)
    y_test = to_one_hot(y_test, dimension=10)
    y_val = to_one_hot(y_val, dimension=10)

    print('\nx_train shape: %s'
          '\ny_train shape: %s'
          '\nx_test shape: %s'
          '\ny_test shape: %s'
          '\nx_val shape: %s'
          '\ny_val shape: %s'
          % (x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape))
    return x_train, y_train, x_test, y_test, x_val, y_val, class_weights_dict

def TrainConv1D(x_train, y_train, x_val, y_val, class_weights_dict, filepth):
    summary    = True
    verbose    = 1
    batch_size = 128
    epochs     = 200
    optimizer  = 'adam'
    lr         = 1e-2
    activator  = 'relu'

    pool_size     = 2
    init_Conv1D   = initializers.lecun_uniform()
    init_Dense    = initializers.he_normal()
    padding_style = 'same'
    drop_rate     = 0.025
    l2_coeff      = 1e-3
    loss_type     = 'categorical_crossentropy'
    metrics       = ('acc',)

    dense_num     = 128
    dropout_dense = 0.25

    if lr > 0:
        if optimizer == 'adam':
            chosed_optimizer = optimizers.Adam(lr=lr)
        elif optimizer == 'sgd':
            chosed_optimizer = optimizers.SGD(lr=lr)
        elif optimizer == 'rmsprop':
            chosed_optimizer = optimizers.RMSprop(lr=lr)

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
        callbacks.ModelCheckpoint(
            filepath=filepth,
            monitor='val_acc',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=verbose,
        )
    ]

    #
    # build model
    #
    network = models.Sequential()
    network.add(layers.SeparableConv1D(filters=16,
                                       kernel_size=5,
                                       activation=activator,
                                       depthwise_initializer=init_Conv1D,
                                       pointwise_initializer=init_Conv1D,
                                       depthwise_regularizer=regularizers.l2(l2_coeff),
                                       pointwise_regularizer=regularizers.l1(l2_coeff),
                                       input_shape=(x_train.shape[1:])))
    network.add(layers.BatchNormalization(axis=-1))
    network.add(layers.Dropout(drop_rate))
    network.add(layers.MaxPooling1D(pool_size=pool_size, padding=padding_style))

    for _ in range(4):
        network.add(layers.SeparableConv1D(filters=32,
                                           kernel_size=5,
                                           activation=activator,
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
                         validation_data=(x_val, y_val),
                         shuffle=True,
                         class_weight=class_weights_dict
                         )
    return network, result.history

if __name__ == '__main__':
    main()
