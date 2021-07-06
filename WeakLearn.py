import sys

import numpy as np
import random
from makedata import Dataset_WPDP
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


def weakLearn(learn_conf,output="./output/weakLearn/"):
    dt = datetime.datetime.now()
    exe_name = "{}_{}_{}_{}{}{}".format(proj,dt.month,dt.day,dt.hour,dt.minute,dt.second)
    learn_conf = {"name":exe_name}


    dataset = Dataset_WPDP(config.dataconf,ohvdir,learn_conf["train"],learn_conf["test"],learn_conf["val"])
    input = dataset.train_data


    #DBNの宣言
    dbn = NeuralNetwork(epochs=50)

    rbms = []

    rbms.append( RBM(input.shape[1:],rdc_axis=1,out_dim=40,lr=learn_conf["lr"],epochs=50,lr_decay=learn_conf["lr_decay"]) )
    rbms.append( RBM(rbms[-1].out_shape,rdc_axis=0,out_dim=40,lr=learn_conf["lr"],epochs=50,lr_decay=learn_conf["lr_decay"]) )

    for rbm in rbms:
        dbn.add(rbm)


    dense = Dense(rbms[-1].out_shape)

    output = LogisticRegression(dense.out_dim,2,lr=lgr_lr,lr_decay=0.95)

    x_t = dbn.predict(dataset.train_data)

    x_v = dense.predict(dbn.predict(dataset.val_data))
    y_v = dataset.val_label

    classifier = FeatureSelectionClassifier(output,x_v,y_v,FS="BFS",epochs=50\
                    ,k=int(dense.out_dim*0.8),dropout_k=int(dense.out_dim*0.98),metrics=auc)

    dbn.add(dense)
    dbn.add(classifier)

    t1 = datetime.datetime.now()
    dbn.fit(dataset.train_data,dataset.train_label)
    t2 = datetime.datetime.now()

    learn_conf["exe_time"] = t2-t1
    learn_conf["DBN"] = dbn

    with open(output+learn_conf["name"], mode="wb") as f:
        pickle.dump(learn_conf, f)

    print("saved",learn_conf["name"])
    for k, v in learn_conf.items():
        print(k,":",v)



if __name__ == "__main__":
    projects = ["Xerces","Xalan","Synapse","poi","Lucene"]
    ohvdir = dsconf["WPDP"][proj]["ohv"]
    trainPRJ = dsconf[proj][dsconf["WPDP"][proj]["val"]]
    testPRJ = dsconf[proj][dsconf["WPDP"][proj]["test"]]
    valPRJ = dsconf[proj][dsconf["WPDP"][proj]["train"]]

    learn_conf["train"] = trainPRJ
    learn_conf["test"] = testPRJ
    learn_conf["val"] = valPRJ

    learn_conf["lr"] = 0.05
    learn_conf["lr_decay"] = 0.95
    learn_conf["clf_lr"] = 0.175
    learn_conf["clf_lr_decay"] = 0.95

    weakLearn(learn_conf)
