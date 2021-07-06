import sys
import os

import numpy as np
import random
from makedata import Dataset_WPDP,addBug
from functions.metrics import accuracy, precision, recall, f_measure, auc
import config
from datasetConfig import dsconf
from makeCPDPohv import CPDPdict

from util.alarm import Sound
from DNN.NeuralNetwork import NeuralNetwork
from DNN.Layer import Dense
from DNN.RBM import RBM
from DNN.LogisticRegression import LogisticRegression
from FeatureSelection import FeatureSelectionClassifier

def CPDPtest(proj,lr_DBN,lr_CLF,layers):
    dict = CPDPdict()
    ohvdir = dict[proj]["ohv"]
    trainPRJ = dict[proj]["train"]
    testPRJ = dict[proj]["test"]
    valPRJ = dict[proj]["val"]

    dataset = Dataset_WPDP(config.dataconf,ohvdir,trainPRJ,testPRJ,valPRJ)

    input = dataset.train_data

    dbn = NeuralNetwork()

    for i,layer in enumerate(layers):
        if i == 0:
            in_shape = input.shape[1:]
        else:
            in_shape = last_layer.out_shape
        last_layer = RBM(in_shape,rdc_axis=layer[0],out_dim=layer[1],lr=lr_DBN,epochs=50,lr_decay=0.95)
        dbn.add(last_layer)

    dbn.fit(dataset.train_data,dataset.train_label)

    lgr = NeuralNetwork(lr=0.1,epochs=50)

    lgr_dense = Dense(last_layer.out_shape)
    lgr_clf = LogisticRegression(lgr_dense.out_dim,2,lr=lr_CLF,lr_decay=0.95)

    lgr.add(lgr_dense)
    lgr.add(lgr_clf)

    clf = NeuralNetwork(lr=0.1,epochs=50)
    dense = Dense(last_layer.out_shape)
    output = LogisticRegression(dense.out_dim,2,lr=lr_CLF,lr_decay=0.95)

    x_t = dbn.predict(dataset.train_data)
    x_v = dense.predict(dbn.predict(dataset.val_data))
    y_v = dataset.val_label

    classifier = FeatureSelectionClassifier(output,x_v,y_v,FS="BFS",epochs=50\
                    ,k=int(dense.out_dim*0.8),dropout_k=int(dense.out_dim*0.98),metrics=auc)

    clf.add(dense)
    clf.add(classifier)

    lgr.fit(x_t,dataset.train_label)
    clf.fit(x_t,dataset.train_label)

    scores = {}

    y_pred = lgr.predict(dense.predict(dbn.predict(dataset.test_data)))
    scores["NoFS"] = {}
    scores["NoFS"]["accuracy"] = accuracy(y_pred,dataset.test_label)
    scores["NoFS"]["precision"] = precision(y_pred,dataset.test_label)
    scores["NoFS"]["recall"] = recall(y_pred,dataset.test_label)
    scores["NoFS"]["f-measure"] = f_measure(y_pred,dataset.test_label)

    y_pred = clf.predict(dense.predict(dbn.predict(dataset.test_data)))
    scores["FS"] = {}
    scores["FS"]["accuracy"] = accuracy(y_pred,dataset.test_label)
    scores["FS"]["precision"] = precision(y_pred,dataset.test_label)
    scores["FS"]["recall"] = recall(y_pred,dataset.test_label)
    scores["FS"]["f-measure"] = f_measure(y_pred,dataset.test_label)

    return scores

def CPDPcoverageValuation(description,lr_DBN,lr_CLF,layers):
    file = "CPDP_{}.csv".format(description)
    name = "prelr-{}_finelr-{}".format(lr_DBN,lr_CLF)
    for l in layers:
        name+="_{}-{}".format(l[0],l[1])
    row = [name,"NoFS_accuracy","NoFS_precision","NoFS_recall","NoFS_F1","FS_accuracy","FS_precision","FS_recall","FS_F1"]
    if os.path.exists(file):
        print("the file is already exists.")
        exit(1)
    with open(file,mode="w") as f:
        f.write(','.join(row)+"\n")
    dict = CPDPdict()
    for proj in dict.keys():
        scores = CPDPtest(proj,lr_DBN,lr_CLF,layers)
        column = [proj] + [str(scores["NoFS"][key]) for key in scores["NoFS"].keys()] + [str(scores["FS"][key]) for key in scores["FS"].keys()]
        print(column)
        with open(file,mode="a") as f:
            f.write(','.join(column)+"\n")

if __name__ == "__main__":
    np.seterr(all='ignore')
    CPDPcoverageValuation("Layer2_clf008",0.15,0.08,[[1,40],[0,40]])
    Sound(3)
