# multi-classification in **NLP** based on Logistic regression, SVM, ConvNet and ResNet
This repo was constructed by ruiyang for the final project of machine learning class at NKU.  
[@ruiyangsong](https://github.com/ruiyangsong/nlp)

## Dependencies
```text
version greater than the listed ones should also work.

|library                   version|
|---------------------------------|
|python                    3.6.9  |
|numpy                     1.16.5 |
|pandas                    0.25.1 |
|matplotlib                3.1.1  |
|scikit-learn              0.21.2 |
|jieba                     0.39   |
|gensim                    3.8.0  |
|tensorflow                1.12.0 |
|keras                     2.2.4  |
```
## Directory tree
```text
nlp:.
│  .gitignore
│  baidu_stopwords.txt
│  README.md
│  userdict.txt
│
├─data
│      data.csv
│      keywords.csv
│      mode_padding.npz
│      mode_stack.npz
│      mode_sum.npz
│      
├─fig
│      data_hist.png
│      data_pie.png
│      roc_auc.png
│      
├─log
│      compare.log
│      convnet_grid_search.log
│      convnet_padding_111_0.01.log
│      logistic_regression_grid_search.log
│      logistic_regression_sum_1000_0.1.log
│      resnet_grid_search.log
│      resnet_padding_81_0.01.log
│      svm_grid_search.log
│      svm_sum_1000_rbf.log
│      
├─model
│  ├─Conv1D
│  │  ├─convnet_mode_padding_epochs_111_lr_0.01_2020.06.12.08.49.55
│  │  │      history.dict
│  │  │      model.json
│  │  │      model.png
│  │  │      test_rst.npz
│  │  │      weightsFinal.h5
│  │  │      
│  │  └─resnet_mode_padding_epochs_81_lr_0.01_2020.06.12.08.50.57
│  │          history.dict
│  │          model.json
│  │          model.png
│  │          test_rst.npz
│  │          weightsFinal.h5
│  │          
│  ├─LR
│  │  └─mode_sum_maxiter_1000_lr_0.1
│  │          test_rst.npz
│  │          thetas.npz
│  │          
│  ├─svm
│  │  └─mode_sum_C_1000.0_kernel_rbf
│  │          test_rst.npz
│  │          
│  └─word2vec
│          dim100_window3_cnt1.model
│          
└─src
    │  compare.py
    │  convnet.py
    │  logistic_regression.py
    │  resnet.py
    │  svm.py
    │  utils.py
    │  word2vec.py            
```
## dataset
Train data set format are as blow, and test data set do not have labels (商品编码).

|样本编号|商品名称|商品价格|商品编码|
|:----:  |:----:  |:----:  |:----:  |
|1|贝蒂斯双瓶礼盒橄榄油|42|101|
|2|充电强光灯灯珠|33|101|
|...|...|...|...|
|12238|转售电力收入|77|110|

![alt text](./fig/data_hist.png 'histogram')
>hist

![alt text](./fig/data_pie.png 'pie chart')
>pie
>
## Usage
Before sinking here, make sure all the dependencies were correctly installed.   
The project is organized by the following
1. split words (implement with jieba for Chinese words)
2. construct word vectors (based on skip-gram, implement with gensim) 
3. select keywords by tf-idf weights
4. generate feature tensors
5. train and evaluate classiers
### clone the repo and change directory to src
```shell script
git clone https://github.com/ruiyangsong/nlp.git
cd nlp/src/
```
### run step 1 to 4
```shell script
python word2vec.py
```
#### Logistic regression
```shell script
python logistic_regression.py sum 1000 0.1
```
#### Support vector machine
```shell script
python svm.py sum 1000 rbf
```
#### ConvNet
```shell script
python convnet.py padding 111 0.01
```
#### ResNet with dilated convolutions
```shell script
python resnet padding 81 0.01
```

## Performance comparision
### The ROC curve and AUC
![alt text](./fig/roc_auc.png 'roc')
>ROC curve

### The predicted scores (LR, SVM, ConvNet, ResNet)
```text
########################################################################################
          |labels    |recall    |precision |F1        |mcc       |F1_micro  |mcc_micro |
LR        ------------------------------------------------------------------------------
          |c1        |0.3395    |0.4670    |0.3932    |0.3359    |0.4065    |0.3405    |
          |c2        |1.0000    |0.7391    |0.8500    |0.8392    |          |          |
          |c3        |0.0000    |0.0000    |0.0000    |0.0000    |          |          |
          |c4        |0.3466    |0.4350    |0.3858    |0.3269    |          |          |
          |c5        |0.0000    |0.0000    |0.0000    |0.0000    |          |          |
          |c6        |0.0000    |0.0000    |0.0000    |0.0000    |          |          |
          |c7        |0.0258    |0.2051    |0.0458    |0.0300    |          |          |
          |c8        |0.0000    |0.0000    |0.0000    |-0.0095   |          |          |
          |c9        |0.8282    |0.2281    |0.3577    |0.2202    |          |          |
          |c10       |1.0000    |0.9655    |0.9825    |0.9811    |          |          |
SVM       ------------------------------------------------------------------------------
          |c1        |0.9779    |0.8466    |0.9075    |0.8981    |0.5617    |0.5130    |
          |c2        |1.0000    |0.9931    |0.9966    |0.9961    |          |          |
          |c3        |0.1512    |0.2653    |0.1926    |0.1558    |          |          |
          |c4        |0.6534    |0.5359    |0.5889    |0.5400    |          |          |
          |c5        |0.6704    |0.4110    |0.5096    |0.4776    |          |          |
          |c6        |0.1014    |0.1829    |0.1304    |0.0957    |          |          |
          |c7        |0.1935    |0.2429    |0.2154    |0.1171    |          |          |
          |c8        |0.4504    |0.4910    |0.4698    |0.4149    |          |          |
          |c9        |0.3359    |0.3275    |0.3316    |0.2031    |          |          |
          |c10       |1.0000    |0.9949    |0.9975    |0.9972    |          |          |
ConvNet   ------------------------------------------------------------------------------
          |c1        |0.1550    |0.9545    |0.2667    |0.3638    |0.3734    |0.3037    |
          |c2        |1.0000    |0.7769    |0.8744    |0.8643    |          |          |
          |c3        |0.2384    |0.1990    |0.2169    |0.1527    |          |          |
          |c4        |0.2988    |0.6818    |0.4155    |0.4142    |          |          |
          |c5        |0.4972    |0.2673    |0.3477    |0.2959    |          |          |
          |c6        |0.5000    |0.0920    |0.1555    |0.0927    |          |          |
          |c7        |0.0742    |0.1933    |0.1072    |0.0453    |          |          |
          |c8        |0.3058    |0.3203    |0.3129    |0.2395    |          |          |
          |c9        |0.0282    |0.3333    |0.0520    |0.0556    |          |          |
          |c10       |1.0000    |1.0000    |1.0000    |1.0000    |          |          |
ResNet    ------------------------------------------------------------------------------
          |c1        |0.6089    |0.7534    |0.6735    |0.6421    |0.4485    |0.3873    |
          |c2        |0.3253    |1.0000    |0.4909    |0.5462    |          |          |
          |c3        |0.5581    |0.2060    |0.3009    |0.2575    |          |          |
          |c4        |0.5339    |0.6837    |0.5996    |0.5652    |          |          |
          |c5        |0.8547    |0.5134    |0.6415    |0.6297    |          |          |
          |c6        |0.2500    |0.0898    |0.1321    |0.0554    |          |          |
          |c7        |0.2742    |0.2640    |0.2690    |0.1607    |          |          |
          |c8        |0.2562    |0.5905    |0.3573    |0.3487    |          |          |
          |c9        |0.1949    |0.5468    |0.2873    |0.2598    |          |          |
          |c10       |1.0000    |0.9949    |0.9975    |0.9972    |          |          |
########################################################################################
```