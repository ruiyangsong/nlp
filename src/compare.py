#!/usr/bin/env/ python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from utils import test_score

def compare_metrics():
    p_reals = []
    p_preds = []
    classifiers = {'LR': '../model/LR/mode_sum_maxiter_1000_lr_0.1/test_rst.npz',
                   'SVM': '../model/svm/mode_sum_C_1000.0_kernel_rbf/test_rst.npz',
                   'ConvNet': '../model/Conv1D/convnet_mode_padding_epochs_111_lr_0.01_2020.06.12.08.49.55/test_rst.npz',
                   'ResNet': '../model/Conv1D/resnet_mode_padding_epochs_81_lr_0.01_2020.06.12.08.50.57/test_rst.npz'}
    for clf, rst_pth in classifiers.items():
        rst_data = np.load(rst_pth)
        y_real   = rst_data['y_real']
        y_pred   = rst_data['y_pred']
        p_reals.append(rst_data['p_real'])
        p_preds.append(rst_data['p_pred'])
        acc_micro, f1_micro, mcc_micro, recalls, precisions, f1s, mccs = test_score(y_real, y_pred)
        classifiers[clf]=(acc_micro, f1_micro, mcc_micro, recalls, precisions, f1s, mccs)

    print('########################################################################################'
          '\n          |labels    |recall    |precision |F1        |mcc       |F1_micro  |mcc_micro |'
          '\nLR        ------------------------------------------------------------------------------'
          '\n          |c1        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|%-10.4f|%-10.4f|'
          '\n          |c2        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c3        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c4        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c5        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c6        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c7        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c8        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c9        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c10       |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\nSVM       ------------------------------------------------------------------------------'
          '\n          |c1        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|%-10.4f|%-10.4f|'
          '\n          |c2        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c3        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c4        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c5        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c6        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c7        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c8        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c9        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c10       |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\nConvNet   ------------------------------------------------------------------------------'
          '\n          |c1        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|%-10.4f|%-10.4f|'
          '\n          |c2        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c3        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c4        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c5        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c6        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c7        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c8        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c9        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c10       |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\nResNet    ------------------------------------------------------------------------------'
          '\n          |c1        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|%-10.4f|%-10.4f|'
          '\n          |c2        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c3        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c4        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c5        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c6        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c7        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c8        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c9        |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n          |c10       |%-10.4f|%-10.4f|%-10.4f|%-10.4f|          |          |'
          '\n########################################################################################'
          % (classifiers['LR'][3][0], classifiers['LR'][4][0], classifiers['LR'][5][0], classifiers['LR'][6][0],classifiers['LR'][1],classifiers['LR'][2],
             classifiers['LR'][3][1], classifiers['LR'][4][1], classifiers['LR'][5][1], classifiers['LR'][6][1],
             classifiers['LR'][3][2], classifiers['LR'][4][2], classifiers['LR'][5][2], classifiers['LR'][6][2],
             classifiers['LR'][3][3], classifiers['LR'][4][3], classifiers['LR'][5][3], classifiers['LR'][6][3],
             classifiers['LR'][3][4], classifiers['LR'][4][4], classifiers['LR'][5][4], classifiers['LR'][6][4],
             classifiers['LR'][3][5], classifiers['LR'][4][5], classifiers['LR'][5][5], classifiers['LR'][6][5],
             classifiers['LR'][3][6], classifiers['LR'][4][6], classifiers['LR'][5][6], classifiers['LR'][6][6],
             classifiers['LR'][3][7], classifiers['LR'][4][7], classifiers['LR'][5][7], classifiers['LR'][6][7],
             classifiers['LR'][3][8], classifiers['LR'][4][8], classifiers['LR'][5][8], classifiers['LR'][6][8],
             classifiers['LR'][3][9], classifiers['LR'][4][9], classifiers['LR'][5][9], classifiers['LR'][6][9],
             classifiers['SVM'][3][0], classifiers['SVM'][4][0], classifiers['SVM'][5][0], classifiers['SVM'][6][0], classifiers['SVM'][1], classifiers['SVM'][2],
             classifiers['SVM'][3][1], classifiers['SVM'][4][1], classifiers['SVM'][5][1], classifiers['SVM'][6][1],
             classifiers['SVM'][3][2], classifiers['SVM'][4][2], classifiers['SVM'][5][2], classifiers['SVM'][6][2],
             classifiers['SVM'][3][3], classifiers['SVM'][4][3], classifiers['SVM'][5][3], classifiers['SVM'][6][3],
             classifiers['SVM'][3][4], classifiers['SVM'][4][4], classifiers['SVM'][5][4], classifiers['SVM'][6][4],
             classifiers['SVM'][3][5], classifiers['SVM'][4][5], classifiers['SVM'][5][5], classifiers['SVM'][6][5],
             classifiers['SVM'][3][6], classifiers['SVM'][4][6], classifiers['SVM'][5][6], classifiers['SVM'][6][6],
             classifiers['SVM'][3][7], classifiers['SVM'][4][7], classifiers['SVM'][5][7], classifiers['SVM'][6][7],
             classifiers['SVM'][3][8], classifiers['SVM'][4][8], classifiers['SVM'][5][8], classifiers['SVM'][6][8],
             classifiers['SVM'][3][9], classifiers['SVM'][4][9], classifiers['SVM'][5][9], classifiers['SVM'][6][9],
             classifiers['ConvNet'][3][0], classifiers['ConvNet'][4][0], classifiers['ConvNet'][5][0], classifiers['ConvNet'][6][0], classifiers['ConvNet'][1], classifiers['ConvNet'][2],
             classifiers['ConvNet'][3][1], classifiers['ConvNet'][4][1], classifiers['ConvNet'][5][1], classifiers['ConvNet'][6][1],
             classifiers['ConvNet'][3][2], classifiers['ConvNet'][4][2], classifiers['ConvNet'][5][2], classifiers['ConvNet'][6][2],
             classifiers['ConvNet'][3][3], classifiers['ConvNet'][4][3], classifiers['ConvNet'][5][3], classifiers['ConvNet'][6][3],
             classifiers['ConvNet'][3][4], classifiers['ConvNet'][4][4], classifiers['ConvNet'][5][4], classifiers['ConvNet'][6][4],
             classifiers['ConvNet'][3][5], classifiers['ConvNet'][4][5], classifiers['ConvNet'][5][5], classifiers['ConvNet'][6][5],
             classifiers['ConvNet'][3][6], classifiers['ConvNet'][4][6], classifiers['ConvNet'][5][6], classifiers['ConvNet'][6][6],
             classifiers['ConvNet'][3][7], classifiers['ConvNet'][4][7], classifiers['ConvNet'][5][7], classifiers['ConvNet'][6][7],
             classifiers['ConvNet'][3][8], classifiers['ConvNet'][4][8], classifiers['ConvNet'][5][8], classifiers['ConvNet'][6][8],
             classifiers['ConvNet'][3][9], classifiers['ConvNet'][4][9], classifiers['ConvNet'][5][9], classifiers['ConvNet'][6][9],
             classifiers['ResNet'][3][0], classifiers['ResNet'][4][0], classifiers['ResNet'][5][0], classifiers['ResNet'][6][0], classifiers['ResNet'][1], classifiers['ResNet'][2],
             classifiers['ResNet'][3][1], classifiers['ResNet'][4][1], classifiers['ResNet'][5][1], classifiers['ResNet'][6][1],
             classifiers['ResNet'][3][2], classifiers['ResNet'][4][2], classifiers['ResNet'][5][2], classifiers['ResNet'][6][2],
             classifiers['ResNet'][3][3], classifiers['ResNet'][4][3], classifiers['ResNet'][5][3], classifiers['ResNet'][6][3],
             classifiers['ResNet'][3][4], classifiers['ResNet'][4][4], classifiers['ResNet'][5][4], classifiers['ResNet'][6][4],
             classifiers['ResNet'][3][5], classifiers['ResNet'][4][5], classifiers['ResNet'][5][5], classifiers['ResNet'][6][5],
             classifiers['ResNet'][3][6], classifiers['ResNet'][4][6], classifiers['ResNet'][5][6], classifiers['ResNet'][6][6],
             classifiers['ResNet'][3][7], classifiers['ResNet'][4][7], classifiers['ResNet'][5][7], classifiers['ResNet'][6][7],
             classifiers['ResNet'][3][8], classifiers['ResNet'][4][8], classifiers['ResNet'][5][8], classifiers['ResNet'][6][8],
             classifiers['ResNet'][3][9], classifiers['ResNet'][4][9], classifiers['ResNet'][5][9], classifiers['ResNet'][6][9],
             ))

    return p_reals, p_preds, list(classifiers.keys())

def auc(p_reals, p_preds, classifier_name, outdir='../fig'):
    # calc and plot micro ROC (AUC)
    plt.figure(figsize=(5,5), dpi=200)
    for i in range(len(p_reals)):
        classifier = classifier_name[i]
        fpr, tpr, thresholds = metrics.roc_curve(p_reals[i].ravel(), p_preds[i].ravel())
        auc = metrics.auc(fpr, tpr)
        # plot
        plt.plot(fpr, tpr, label='%s_auc: %.2f' % (classifier,auc))
    plt.plot((0, 1), (0, 1),linestyle='--')  # diagonal dash line
    plt.xlim((-0.01, 1.01))
    plt.ylim((-0.01, 1.01))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.title('ROC curve and AUC')
    # plt.show()
    plt.savefig('%s/roc_auc.png' % outdir)  # pngfile

if __name__ == '__main__':
    p_reals, p_preds, clfs = compare_metrics()
    auc(p_reals, p_preds, clfs)