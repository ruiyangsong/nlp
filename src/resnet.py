#!/usr/bin/env python
import os, sys, time
import numpy as np
from keras import Input, models, layers, optimizers, callbacks, regularizers, initializers
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
    model, history_dict = TrainResNet(x_train, y_train, x_val, y_val, class_weights_dict, filepth)
    net_saver(model, modeldir, history_dict)

    #
    # test and save
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

def TrainResNet(x_train, y_train, x_val, y_val, class_weights_dict, filepth):
    summary    = True
    verbose    = 1
    batch_size = 128
    epochs     = 200
    optimizer  = 'adam'
    lr         = 1e-2
    activator  = 'relu'

    kernel_size   = 3
    pool_size     = 2
    init_Conv1D   = initializers.lecun_uniform()
    init_Dense    = initializers.he_normal()
    padding_style = 'same'
    dropout_rate  = 0.025
    l2_coeff      = 1e-3
    loss_type     = 'categorical_crossentropy'
    metrics       = ('acc',)

    ## used in the dilation loop
    dilation_lower        = 1
    dilation_upper        = 16
    dilation1D_layers     = 16
    dilation1D_filter_num = 16

    ## used in the reduce loop
    residual_stride     = 2
    reduce_layers       = 6  # 300 -> 150 -> 75 -> 38 -> 19 -> 10 -> 5
    reduce1D_filter_num = 16

    dense_num     = 128
    dropout_dense = 0.25


    if lr > 0:
        if optimizer == 'adam':
            chosed_optimizer = optimizers.Adam(lr=lr)
        elif optimizer == 'sgd':
            chosed_optimizer = optimizers.SGD(lr=lr)
        elif optimizer == 'rmsprop':
            chosed_optimizer = optimizers.RMSprop(lr=lr)

    #
    # callback
    #
    my_callbacks = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            verbose=verbose
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
    # build
    #
    ## basic Conv1D
    input_layer = Input(shape=x_train.shape[1:])

    y = layers.SeparableConv1D(
        filters=dilation1D_filter_num,
        kernel_size=1,
        padding=padding_style,
        kernel_initializer=init_Conv1D,
        activation=activator)(input_layer)
    res = layers.BatchNormalization(axis=-1)(y)

    ## loop with Conv1D with dilation (padding='same')
    for _ in range(dilation1D_layers):
        y = layers.SeparableConv1D(
            filters=dilation1D_filter_num,
            kernel_size=kernel_size,
            padding=padding_style,
            dilation_rate=dilation_lower,
            kernel_initializer=init_Conv1D,
            activation=activator,
            kernel_regularizer=regularizers.l2(l2_coeff))(res)
        y = layers.BatchNormalization(axis=-1)(y)
        y = layers.Dropout(dropout_rate)(y)
        y = layers.SeparableConv1D(
            filters=dilation1D_filter_num,
            kernel_size=kernel_size,
            padding=padding_style,
            dilation_rate=dilation_lower,
            kernel_initializer=init_Conv1D,
            activation=activator,
            kernel_regularizer=regularizers.l2(l2_coeff))(y)
        y = layers.BatchNormalization(axis=-1)(y)

        res = layers.add([y, res])

        dilation_lower *= 2
        if dilation_lower > dilation_upper:
            dilation_lower = 1

    ## residual block to reduce dimention.
    for _ in range(reduce_layers):
        y = layers.SeparableConv1D(
            filters=reduce1D_filter_num,
            kernel_size=kernel_size,
            padding=padding_style,
            kernel_initializer=init_Conv1D,
            activation=activator,
            kernel_regularizer=regularizers.l2(l2_coeff))(res)
        y = layers.BatchNormalization(axis=-1)(y)
        y = layers.Dropout(dropout_rate)(y)
        y = layers.MaxPooling1D(pool_size, padding=padding_style)(y)
        res = layers.SeparableConv1D(
            filters=reduce1D_filter_num,
            kernel_size=kernel_size,
            strides=residual_stride,
            padding=padding_style,
            kernel_initializer=init_Conv1D,
            activation=activator,
            kernel_regularizer=regularizers.l2(l2_coeff))(res)
        res = layers.add([y, res])

    ## flat & dense
    y = layers.Flatten()(y)
    y = layers.Dense(dense_num, activation=activator)(y)
    y = layers.BatchNormalization(axis=-1)(y)
    y = layers.Dropout(dropout_dense)(y)

    output_layer = layers.Dense(10,activation='softmax')(y)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    if summary:
        model.summary()

    model.compile(optimizer=chosed_optimizer,
                  loss=loss_type,
                  metrics=list(metrics)  # accuracy
                  )

    # K.set_session(tf.Session(graph=model.output.graph))
    # init = K.tf.global_variables_initializer()
    # K.get_session().run(init)

    result = model.fit(x=x_train,
                       y=y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       callbacks=my_callbacks,
                       validation_data=(x_val, y_val),
                       shuffle=True,
                       class_weight=class_weights_dict
                       )
    return model, result.history


if __name__ == '__main__':
    main()





