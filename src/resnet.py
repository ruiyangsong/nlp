#!/usr/bin/env python
import os, sys, time
import numpy as np
from keras import Input, models, layers, optimizers, callbacks, regularizers, initializers
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
                model, history_dict = TrainResNet(x_train, y_train, x_val, y_val, class_weights_dict, filepth=None, epochs=epochs, lr=learning_rate,verbose=verbose)
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
    model_name = '%s_mode_%s_epochs_%s_lr_%s_%s'%(sys.argv[0].split('/')[-1][:-3],
                                                  MODE,EPOCHS,LEARNING_RATE,
                                                  time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime()))
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
    model, history_dict = TrainResNet(x_train, y_train, class_weights_dict=class_weights_dict, epochs=EPOCHS, lr=LEARNING_RATE)
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

def TrainResNet(x_train, y_train, x_val=None, y_val=None, class_weights_dict=None, filepth=None, epochs=200, lr=1e-2, verbose=1):
    summary    = True
    batch_size = 128
    optimizer  = 'adam'
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
    reduce_layers       = 6  # 100 -> 50 -> 25 -> 13 -> 7 -> 4 -> 2
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
                       validation_data=val_data,
                       shuffle=True,
                       class_weight=class_weights_dict
                       )
    return model, result.history


if __name__ == '__main__':
    if len(sys.argv) == 1:
        grid_search()
    else:
        main()





