import sys

import numpy as np
import random
from makedata import Dataset_WPDP,addBug
from functions.metrics import accuracy, precision, recall, f_measure, auc
import config
from datasetConfig import dsconf

from util.alarm import Sound
from DNN.NeuralNetwork import NeuralNetwork
from DNN.Layer import Dense
from DNN.RBM import RBM
from DNN.LogisticRegression import LogisticRegression
from FeatureSelection import FeatureSelectionClassifier

#テスト用
import pickle
import datetime
import time

np.seterr(all='ignore')
#dict.csvとsequence.csvを入れるディレクトリ名
#githubohvの下に置く

def test(proj):
    dt = datetime.datetime.now()
    exelog = {"name":"{}_{}_{}_{}{}{}".format(proj,dt.month,dt.day,dt.hour,dt.minute,dt.second)}
    #訓練データ,検証データ,テストデータの読み込み
    ohvdir = dsconf["WPDP"][proj]["ohv"]
    trainPRJ = dsconf[proj][dsconf["WPDP"][proj]["val"]]
    testPRJ = dsconf[proj][dsconf["WPDP"][proj]["test"]]
    valPRJ = dsconf[proj][dsconf["WPDP"][proj]["train"]]
    config.dataconf["threshold"] = 1
    dataset = Dataset_WPDP(config.dataconf,ohvdir,trainPRJ,testPRJ,valPRJ)
    input = dataset.train_data

    #DBNの宣言
    dbn = NeuralNetwork(lr=0.1,epochs=50)

    outshape = [40,40]
    node_num = 2
    dim_num = 3
    nodes = [outshape[-2] + int((input.shape[-2]-outshape[-2]) *(i/node_num)) for i in reversed(range(node_num))]
    dims =  [outshape[-1] + int((input.shape[-1]-outshape[-1]) *(i/dim_num)) for i in reversed(range(dim_num))]
    #print(nodes,dims)
    rbms = []
    lr = 0.05
    lr_decay = 0.95
    rbms.append( RBM(input.shape[1:],rdc_axis=1,out_dim=40,lr=lr,epochs=50,lr_decay=lr_decay) )
    #rbms.append( RBM(rbms[-1].out_shape,rdc_axis=1,out_dim=50,lr=lr,epochs=50,lr_decay=lr_decay) )
    #rbms.append( RBM(rbms[-1].out_shape,rdc_axis=1,out_dim=40,lr=lr,epochs=50,lr_decay=lr_decay) )
    #rbms.append( RBM(rbms[-1].out_shape,rdc_axis=1,out_dim=40,lr=lr,epochs=50,lr_decay=lr_decay) )
    #rbms.append( RBM(rbms[-1].out_shape,rdc_axis=0,out_dim=100,lr=lr,epochs=50,lr_decay=lr_decay) )
    #rbms.append( RBM(rbms[-1].out_shape,rdc_axis=0,out_dim=100,lr=lr,epochs=50,lr_decay=lr_decay) )
    #rbms.append( RBM(rbms[-1].out_shape,rdc_axis=0,out_dim=100,lr=lr,epochs=50,lr_decay=lr_decay) )
    rbms.append( RBM(rbms[-1].out_shape,rdc_axis=0,out_dim=40,lr=lr,epochs=50,lr_decay=lr_decay) )




    for rbm in rbms:
        dbn.add(rbm)
    print(dataset.train_data.shape)
    #DBNの学習
    dbn.fit(dataset.train_data,dataset.train_label)

    lgr_lr = 0.175

    #分類器
    lgr = NeuralNetwork(lr=0.1,epochs=50)

    lgr_dense = Dense(rbms[-1].out_shape)
    lgr_clf = LogisticRegression(lgr_dense.out_dim,2,lr=lgr_lr,lr_decay=0.95)

    lgr.add(lgr_dense)
    lgr.add(lgr_clf)

    clf = NeuralNetwork(lr=0.1,epochs=50)

    dense = Dense(rbms[-1].out_shape)

    output = LogisticRegression(dense.out_dim,2,lr=lgr_lr,lr_decay=0.95)

    x_t = dbn.predict(dataset.train_data)


    #x_v = dense.predict(dbn.predict(dataset.test_data))
    #y_v = dataset.test_label

    x_v = dense.predict(dbn.predict(dataset.val_data))
    y_v = dataset.val_label

    classifier = FeatureSelectionClassifier(output,x_v,y_v,FS="BFS",epochs=50\
                    ,k=int(dense.out_dim*0.8),dropout_k=int(dense.out_dim*0.98),metrics=auc)

    clf.add(dense)
    clf.add(classifier)


    lgr.fit(x_t,dataset.train_label)
    clf.fit(x_t,dataset.train_label)

    print("No Feature Selection")
    y_pred = lgr.predict(dense.predict(dbn.predict(dataset.test_data)))
    print(len(y_pred),len(y_pred[y_pred[:,1] > 0.5]))
    print('accuracy',accuracy(y_pred,dataset.test_label))
    p = precision(y_pred,dataset.test_label)
    print('precision',p)
    r = recall(y_pred,dataset.test_label)
    print('recall',r)
    print('f-measure',2*p*r/(p+r))


    print("Feature Selection")
    y_pred = clf.predict(dense.predict(dbn.predict(dataset.test_data)))
    print(len(y_pred),len(y_pred[y_pred[:,1] > 0.5]))
    print('accuracy',accuracy(y_pred,dataset.test_label))
    p = precision(y_pred,dataset.test_label)
    print('precision',p)
    r = recall(y_pred,dataset.test_label)
    print('recall',r)
    print('f-measure',2*p*r/(p+r))


if __name__ == "__main__":
    projects = ["Xerces","Xalan","Synapse","poi","Lucene"]
    for proj in projects:
        test(proj)
    Sound(3)
