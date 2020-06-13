import os, time
import numpy as np
import pandas as pd

def data_pie(datapth, outdir='../fig'):
    import matplotlib.pyplot as plt
    os.makedirs(outdir,exist_ok=True)
    plt.figure(dpi=200)
    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
    df = pd.read_csv(datapth, encoding='utf-8').loc[:, '商品编码'].value_counts()
    labels = list(df.index)
    cnts = list(df)

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(cnts, autopct=lambda pct: func(pct, cnts), textprops=dict(color="w"))
    ax.legend(wedges, labels,
              title="Labels",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=8, weight="bold", color='black')
    ax.set_title('Train data landscape: A pie')
    # plt.show()
    plt.savefig('%s/data_pie.png'%outdir)  # pngfile

def data_hist(datapth, outdir='../fig'):
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    plt.figure(dpi=200)
    df = pd.read_csv(datapth,encoding='utf-8').loc[:,'商品编码'].value_counts()
    x = np.arange(len(df))+1
    y = list(df)
    xtick = list(df.index)
    plt.bar(x, y, width=0.35, align='center', color='c')
    plt.xticks(x, xtick, rotation=0)
    for a, b in zip(x, y):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    plt.title('Train data landscape: A histogram')
    plt.xlabel('labels')
    plt.ylabel('counts')
    plt.grid(True,axis='y')
    # plt.show()
    plt.savefig('%s/data_hist.png'%outdir)  # pngfile

def load_iris_data():
    '''This func just for debug'''
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, header=None, names=[
        "Sepal length (cm)",
        "Sepal width (cm)",
        "Petal length (cm)",
        "Petal width (cm)",
        "Species"
    ])
    print(df.head())

def conf_mat(y_real, y_pred):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_real, y_pred)

def _metric(cm, classes):
    if classes is None:
        classes = cm.shape[0]
    # print(cm)
    tps = [cm[i, i] for i in range(classes)]
    fns = [np.sum(cm[i, :]) - cm[i, i] for i in range(classes)]
    fps = [np.sum(cm[:, i]) - cm[i, i] for i in range(classes)]
    tns = [np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i] for i in range(classes)]
    return tps, tns, fps, fns

def test_score(y_real, y_pred, classes=None):
    '''y_real and y_pred are 0D array'''
    if classes is None:
        classes = len(np.unique(np.hstack((y_real, y_pred))))
    cm = conf_mat(y_real, y_pred)
    tps, tns, fps, fns = _metric(cm, classes)

    recalls    = np.array([tps[i] / (tps[i] + fns[i]) for i in range(classes)])
    precisions = np.array([tps[i] / (tps[i] + fps[i]) for i in range(classes)])
    f1s        = np.array([2 * tps[i] / (2 * tps[i] + fps[i] + fns[i]) for i in range(classes)])
    mccs       = np.array([(tps[i] * tns[i] - fps[i] * fns[i]) /
                           (np.sqrt((tps[i] + fps[i]) * (tps[i] + fns[i]) * (tns[i] + fps[i]) * (tns[i] + fns[i])))
                           for i in range(classes)])

    #########################################################################
    # Note that acc and f1 are equal in this micro_avg multi-clf situation. #
    # score in micro_avg                                                    #
    #########################################################################
    acc       = np.trace(cm) / np.sum(cm)
    precision = sum(tps) / sum(np.add(tps, fps))
    recall    = sum(tps) / sum(np.add(tps, fns))
    f1        = 2 * precision * recall / (precision + recall) # or use: f1 = 2 * sum(tps) / (2 * sum(tps) + sum(fps) + sum(fns))
    mcc       = (sum(tps) * sum(tns) - sum(fps) * sum(fns)) / \
                      np.sqrt(sum(np.add(tps, fps)) * sum(np.add(tps, fns)) * sum(np.add(tns, fps)) * sum(np.add(tns, fns)))

    recalls[np.isnan(recalls)] = 0
    precisions[np.isnan(precisions)] = 0
    f1s[np.isnan(f1s)] = 0
    mccs[np.isnan(mccs)] = 0

    return acc, f1, mcc, recalls, precisions, f1s, mccs

def calc_class_weights(y_train):
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train.reshape(-1))
    class_weights_dict = dict(enumerate(class_weights))
    return class_weights_dict

def to_one_hot(labels, classes=None):
    if classes is None:
        classes = len(np.unique(labels))
    results = np.zeros((len(labels), classes))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

def split_data(x, y, rate=0.2, seed=0, verbose=0):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=rate, random_state=seed)
    if verbose:
        print('x_train shape: %s'
              '\ny_train shape: %s'
              '\nx_[test|val] shape: %s'
              '\ny_[test|val] shape: %s'%(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    return x_train, y_train, x_test, y_test

def config_tf(user_mem=2500, cuda_rate=0.2):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    queueGPU(USER_MEM=user_mem, INTERVAL=60)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if cuda_rate != 'full':
        config = tf.ConfigProto()
        if float(cuda_rate) < 0.1:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = float(cuda_rate)
        set_session(tf.Session(config=config))

def queueGPU(USER_MEM=10000,INTERVAL=60,Verbose=1):
    """
    :param USER_MEM: int, Memory in Mib that your program needs to allocate
    :param INTERVAL: int, Sleep time in second
    :return:
    """
    try:
        totalmemlst=[int(x.split()[2]) for x in os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Total').readlines()]
        assert USER_MEM<=max(totalmemlst)
    except:
        print('\033[1;35m[WARNING]\nUSER_MEM should smaller than one of the GPU_TOTAL --> %s MiB.\nReset USER_MEM to %s MiB.\033[0m'%(totalmemlst,max(totalmemlst)-1))
        USER_MEM=max(totalmemlst)-1
    while True:
        memlst=[int(x.split()[2]) for x in os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').readlines()]
        if Verbose:
            os.system("echo 'Check at:' `date`")
            print('GPU Free Memory List --> %s MiB'%memlst)
        idxlst=sorted(range(len(memlst)), key=lambda k: memlst[k])
        boollst=[y>USER_MEM for y in sorted(memlst)]
        try:
            GPU=idxlst[boollst.index(True)]
            os.environ['CUDA_VISIBLE_DEVICES']=str(GPU)
            print('GPU %s was chosen.'%GPU)
            break
        except:
            time.sleep(INTERVAL)

def loss_plot(history_dict, outpth):
    from matplotlib import pyplot as plt
    loss = history_dict['loss']
    acc = history_dict['acc']
    val_loss = history_dict['val_loss']
    val_acc = history_dict['val_acc']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 5), dpi=100)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    # plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation mcc')
    # plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.grid(True)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(outpth)#pngfile
    plt.clf()

def net_saver(model, modeldir, history_dict):
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

def net_predictor(modeldir, x_test, y_test, Onsave=True):
    from keras import models
    # Load model
    with open('%s/model.json' % modeldir, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = models.model_from_json(loaded_model_json)  # keras.models.model_from_yaml(yaml_string)
    loaded_model.load_weights(filepath='%s/weightsFinal.h5' % modeldir)
    p_real = y_test
    p_pred = loaded_model.predict(x_test, batch_size=32, verbose=0)  # prob ndarray
    y_real = np.argmax(y_test, axis=1)
    y_pred = np.argmax(p_pred, axis=1)  # 0D array

    #
    # save p_pred
    #
    if Onsave:
        try:
            np.savez('%s/test_rst.npz' % modeldir, p_real=p_real, p_pred=p_pred, y_real=y_real, y_pred=y_pred)
        except:
            print('save test_rst failed')

    return y_test, p_pred, y_real, y_pred

if __name__ == '__main__':
    # generate_array(mode='stack')
    # generate_array(mode='sum')
    data_hist('../data/data.csv')
    data_pie('../data/data.csv')
    # pass
