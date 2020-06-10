import os, sys, time
import numpy as np
from keras import Input, models, layers, optimizers, callbacks, regularizers
from utils import split_data, calc_class_weights, to_one_hot, config_tf,loss_plot

def data(data_pth):
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
    summary = False
    verbose = 1

    #
    # setHyperParams
    #
    batch_size = 128  # {{choice([32, 64, 128])}}
    epochs = 200

    optimizer = 'adam'
    # lr = {{loguniform(np.log(1e-5), np.log(1e-2))}}
    lr = 0.001
    activator = 'elu'  # {{choice(['elu', 'relu'])}}

    kernel_size = 3
    pool_size = 2
    initializer = 'random_uniform'
    padding_style = 'same'
    loss_type = 'categorical_crossentropy'
    metrics = ('acc',)

    dilation1D_layers = 32  # {{choice([16,32])}}
    dilation1D_filter_num = 32  # {{choice([16, 32, 64])}}#used in the loop
    dilation_lower = 1
    dilation_upper = 16

    reduce_layers = 4  # conv 3 times: 120 => 60 => 30 => 15 ==> 8
    reduce1D_filter_num = 16  # {{choice([16, 32])}}#used for reduce dimention

    residual_stride = 2

    dense_num = 128  # {{choice([64, 128])}}

    dropout_rate = 0.25  # {{uniform(0.0001, 0.25)}}
    l2_rate = 0.001  # {{uniform(0.0001, 0.01)}}

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
            factor=0.1,
            patience=10,
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-8,
            patience=20,
            verbose=verbose,
        ),
        callbacks.ModelCheckpoint(
            filepath=filepth,
            monitor='val_loss',
            verbose=verbose,
            mode='min',
            save_best_only=True,
            save_weights_only=True
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
        kernel_initializer=initializer,
        activation=activator)(input_layer)
    res = layers.BatchNormalization(axis=-1)(y)

    ## loop with Conv1D with dilation (padding='same')
    for _ in range(dilation1D_layers):
        y = layers.SeparableConv1D(
            filters=dilation1D_filter_num,
            kernel_size=kernel_size,
            padding=padding_style,
            dilation_rate=dilation_lower,
            kernel_initializer=initializer,
            activation=activator,
            kernel_regularizer=regularizers.l2(l2_rate))(res)
        y = layers.BatchNormalization(axis=-1)(y)
        y = layers.Dropout(dropout_rate)(y)
        y = layers.SeparableConv1D(
            filters=dilation1D_filter_num,
            kernel_size=kernel_size,
            padding=padding_style,
            dilation_rate=dilation_lower,
            kernel_initializer=initializer,
            activation=activator,
            kernel_regularizer=regularizers.l2(l2_rate))(y)
        y = layers.BatchNormalization(axis=-1)(y)

        res = layers.add([y, res])

        dilation_lower *= 2
        if dilation_lower > dilation_upper:
            dilation_lower = 1

    ## Conv1D with dilation (padding='valaid') and residual block to reduce dimention.
    for _ in range(reduce_layers):
        y = layers.SeparableConv1D(
            filters=reduce1D_filter_num,
            kernel_size=kernel_size,
            padding=padding_style,
            kernel_initializer=initializer,
            activation=activator,
            kernel_regularizer=regularizers.l2(l2_rate))(res)
        y = layers.BatchNormalization(axis=-1)(y)
        y = layers.Dropout(dropout_rate)(y)
        y = layers.MaxPooling1D(pool_size, padding=padding_style)(y)
        res = layers.SeparableConv1D(
            filters=reduce1D_filter_num,
            kernel_size=kernel_size,
            strides=residual_stride,
            padding=padding_style,
            kernel_initializer=initializer,
            activation=activator,
            kernel_regularizer=regularizers.l2(l2_rate))(res)
        res = layers.add([y, res])

    ## flat & dense
    y = layers.Flatten()(y)
    y = layers.Dense(dense_num, activation=activator)(y)
    y = layers.BatchNormalization(axis=-1)(y)
    y = layers.Dropout(dropout_rate)(y)

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
                       )
    return model, result.history


# def TrainConv1D(x_train, y_train, x_val, y_val, class_weights_dict, filepth):
#     row_num, col_num = x_train.shape[1:3]
#     verbose    = 1
#     summary    = True
#     batch_size = 128
#     epochs     = 200
#     lr         = 1e-2
#     metrics    = ('acc',)
#     padding_style = 'same'
#
#     my_callbacks = [
#         callbacks.ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.33,
#             patience=5,
#         ),
#         callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=10,
#         ),
#         callbacks.ModelCheckpoint(
#             filepath=filepth,
#             monitor='val_loss',
#             verbose=1,
#             save_best_only=True,
#             mode='min',
#             save_weights_only=True)
#     ]
#
#     network = models.Sequential()
#     network.add(layers.SeparableConv1D(filters=32,
#                                        kernel_size=5,
#                                        activation='relu',
#                                        input_shape=(row_num, col_num)))
#     network.add(layers.MaxPooling1D(pool_size=2, padding=padding_style))
#     network.add(layers.SeparableConv1D(filters=32,
#                                        kernel_size=5,
#                                        activation='relu'))
#     network.add(layers.MaxPooling1D(pool_size=2, padding=padding_style))
#     network.add(layers.SeparableConv1D(filters=64,
#                                        kernel_size=3,
#                                        activation='relu'))
#     network.add(layers.MaxPooling1D(pool_size=2, padding=padding_style))
#     network.add(layers.Flatten())
#     network.add(layers.Dense(128, activation='relu'))
#     network.add(layers.Dropout(0.3))
#     network.add(layers.Dense(16, activation='relu'))
#     network.add(layers.Dropout(0.3))
#     network.add(layers.Dense(10, activation='softmax'))
#
#     if summary:
#         print(network.summary())
#
#     adam = optimizers.Adam(lr=lr)
#     network.compile(optimizer=adam,
#                     loss='categorical_crossentropy',
#                     metrics=list(metrics))
#     result = network.fit(x=x_train,
#                          y=y_train,
#                          batch_size=batch_size,
#                          epochs=epochs,
#                          verbose=verbose,
#                          callbacks=my_callbacks,
#                          validation_data=(x_val, y_val),
#                          shuffle=True,
#                          class_weight=class_weights_dict
#                          )
#     return network, result.history

def saver(modeldir):
    ## save model architecture
    try:
        model_json = model.to_json()
        with open('%s/model.json' % modeldir, 'w') as json_file:
            json_file.write(model_json)
    except:
        print('save model.json to json failed.')

    ## save model weights
    try:
        model.save_weights(filepath='%s/weightsFinal.h5' % modeldir)
    except:
        print('save final model weights failed.')

    ## save training history
    try:
        with open('%s/history.dict' % modeldir, 'w') as file:
            file.write(str(history_dict))
        # with open('%s/fold_%s_history.dict'%(modeldir,k_count), 'r') as file:
        #     print(eval(file.read()))
    except:
        print('save history_dict failed.')

    ## save loss figure
    try:
        figure_pth = '%s/lossFigure.png' %modeldir
        loss_plot(history_dict, outpth=figure_pth)
    except:
        print('save loss plot figure failed.')


def test_model(modeldir, x_test, y_test):
    # Load model
    with open('%s/model.json' %modeldir, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = models.model_from_json(loaded_model_json)  # keras.models.model_from_yaml(yaml_string)
    loaded_model.load_weights(filepath='%s/weightsFinal.h5' %modeldir)

    #
    # # Test model
    # #
    # pearson_coeff, std, mae = test_report_reg(loaded_model, x_test, ddg_test)
    # print('\n----------Predict:'
    #       '\npearson_coeff: %s, std: %s, mae: %s'
    #       % (pearson_coeff, std, mae))
    # score_dict['pearson_coeff'].append(pearson_coeff)
    # score_dict['std'].append(std)
    # score_dict['mae'].append(mae)
    #
    # train_score_dict['pearson_coeff'].append(history_dict['pearson_r'][-1])
    # train_score_dict['std'].append(history_dict['rmse'][-1])
    # train_score_dict['mae'].append(history_dict['mean_absolute_error'][-1])
    #
    # k_count += 1
    #
    # #
    # # save score dict
    # #
    # try:
    #     with open('%s/fold_._score.dict' % modeldir, 'w') as file:
    #         file.write(str(score_dict))
    # except:
    #     print('save score dict failed')
    #
    # #
    # # save AVG score
    # #
    # try:
    #     with open('%s/fold_.avg_score_train_test.txt' % modeldir, 'w') as file:
    #         file.writelines('----------train AVG results\n')
    #         for key in score_dict.keys():
    #             file.writelines('*avg(%s): %s\n' % (key, np.mean(train_score_dict[key])))
    #         file.writelines('----------test AVG results\n')
    #         for key in score_dict.keys():
    #             file.writelines('*avg(%s): %s\n' % (key, np.mean(score_dict[key])))
    # except:
    #     print('save AVG score failed')
    #
    # print('\nAVG results', '-' * 10)
    # for key in score_dict.keys():
    #     print('*avg(%s): %s' % (key, np.mean(score_dict[key])))

if __name__ == '__main__':
    #
    # load data
    #
    data_pth = '../data/mode_sum.npz'
    x_train, y_train, x_test, y_test, x_val, y_val, class_weights_dict = data(data_pth)

    #
    # train
    #
    config_tf(user_mem=2500, cuda_rate=0.2)
    modeldir = '../model/%s/%s' % (sys.argv[0].split('/')[-1][:-3], time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime()))
    os.makedirs(modeldir, exist_ok=True)
    filepth = '%s/weights-best.h5' % modeldir
    model, history_dict = TrainConv1D(x_train, y_train, x_val, y_val, class_weights_dict, filepth)

    #
    # saver
    #
    saver(modeldir)





