import sys

import os
import numpy as np
import random
from makedata import Dataset_WPDP
from functions.metrics import accuracy, precision, recall, f_measure,  auc
import config
from datasetConfig import dsconf


from makeohv import getOhvDirName
from DNN.NeuralNetwork import NeuralNetwork
from DNN.Layer import Dense
from DNN.RBM import RBM
from DNN.LogisticRegression import LogisticRegression
from FeatureSelection import FeatureSelectionClassifier


"""
dataconf
DNNconf
datasetconf
DPtype      "WPDP"or"CPDP"
proj  DPtypeが"WPDP"のときプロジェクト名、"CPDP"のとき??
"""
def testParameter(dataconf,DNNconf,dsconf,DPtype="WPDP",proj="Xerces",fixed=[""]):
    if DPtype == "WPDP":
        vers = dsconf["WPDP"][proj]
        ohvdir = dsconf["WPDP"][proj]["ohv"]
        trainPRJ = dsconf[proj][vers["train"]]
        testPRJ = dsconf[proj][vers["test"]]
        valPRJ = dsconf[proj][vers["val"]]
    elif DPtype == "CPDP":
        ohvdir = dsconf["CPDP"][proj]["ohv"]
        trainPRJ = dsconf[dsconf["CPDP"][proj]["train"][0]][dsconf["CPDP"][proj]["train"][1]]
        testPRJ = dsconf[dsconf["CPDP"][proj]["test"][0]][dsconf["CPDP"][proj]["test"][1]]
        valPRJ = dsconf[dsconf["CPDP"][proj]["val"][0]][dsconf["CPDP"][proj]["val"][1]]
    else:
        print("欠陥検知の設定DPtypeが間違っている >",DPtype)
        exit(0)

    #fixedに入っている要素は固定して評価をとる
    #固定する際は最初の要素で固定する
    for f in fixed:
        DNNconf[f] = [ DNNconf[f][0] ]

    for thr in DNNconf["threshold"]:
        dataconf["threshold"] = thr
        dataset = Dataset_WPDP(dataconf,ohvdir,trainPRJ,testPRJ,valPRJ)
        input = dataset.train_data





        outshape = [40,40]
        for lr in DNNconf["LearningRate"]:
            for dim_type in DNNconf["Dimension"]:
                for node_num in DNNconf["HiddenLayer"]:
                    rbms = []
                    dbn = NeuralNetwork(epochs=50)
                    lastshape = input.shape[1:]
                    for dim_reduc in range(node_num):
                        if dim_reduc == node_num -1:
                            rbms.append(RBM(lastshape,rdc_axis=1,out_dim=outshape[-1],lr=lr,epochs=50,lr_decay=0.95))
                        else:
                            if dim_type == "Linear":
                                _dim =  outshape[-1] + int((input.shape[-1]-outshape[-1]) *((node_num-dim_reduc-1)/node_num))
                            else:
                                _dim = int(dim_type)
                            rbms.append(RBM(lastshape,rdc_axis=1,out_dim=_dim,lr=lr,epochs=50,lr_decay=0.95))
                        lastshape = rbms[-1].out_shape
                    for node_reduc in range(node_num):
                        if node_reduc == node_num -1:
                            rbms.append(RBM(lastshape,rdc_axis=0,out_dim=outshape[-2],lr=lr,epochs=50,lr_decay=0.95))
                        else:
                            if dim_type == "Linear":
                                _nodes =  outshape[-2] + int((input.shape[-2]-outshape[-2]) *((node_num-node_reduc-1)/node_num))
                            else:
                                _nodes = int(dim_type)
                            rbms.append(RBM(lastshape,rdc_axis=0,out_dim=_nodes,lr=lr,epochs=50,lr_decay=0.95))
                        lastshape = rbms[-1].out_shape
                    for rbm in rbms:
                        dbn.add(rbm)
                    dbn.fit(dataset.train_data,dataset.train_label)
                    lgr = NeuralNetwork(lr=0.15,epochs=50)
                    lgr_dense = Dense(rbms[-1].out_shape)
                    lgr_clf = LogisticRegression(lgr_dense.out_dim,2,lr=0.15,lr_decay=0.95)

                    lgr.add(lgr_dense)
                    lgr.add(lgr_clf)

                    clf = NeuralNetwork(lr=0.15,epochs=50)
                    dense = Dense(rbms[-1].out_shape)

                    output = LogisticRegression(dense.out_dim,2,lr=0.15,lr_decay=0.95)

                    x_t = dbn.predict(dataset.train_data)
                    v_pred = dbn.predict(dataset.val_data)
                    print(np.mean(v_pred))
                    x_v = dense.predict(v_pred)
                    y_v = dataset.val_label

                    classifier = FeatureSelectionClassifier(output,x_v,y_v,FS="BFS",epochs=50,\
                                                                k=int(dense.out_dim*0.8),dropout_k=int(dense.out_dim*0.98),metrics=auc)

                    clf.add(dense)
                    clf.add(classifier)

                    lgr.fit(x_t,dataset.train_label)
                    clf.fit(x_t,dataset.train_label)
                    if not os.path.exists("./output/"+proj):
                        os.mkdir("./output/"+proj)
                    with open("./output/{}/{}_lr-{}_layer-{}_htype-{}.txt".format(proj,proj,lr,node_num,dim_type),mode="w") as wf:
                        description = "lr-{},layer-{},htype-{}.txt".format(lr,node_num,dim_type)
                        title = "  FS,Presicion,Recall   ,F-measure "
                        eval = "{:>4},{:>.7f},{:>.7f},{:>.7f}"

                        print("\n",description)
                        print(title)
                        wf.write(description+"\n")
                        wf.write(title+"\n")
                        dbn_predict = dbn.predict(dataset.test_data)
                        y_pred = lgr.predict(dbn_predict)
                        p = precision(y_pred,dataset.test_label)
                        r = recall(y_pred,dataset.test_label)
                        f = f_measure(y_pred,dataset.test_label)
                        print(eval.format("NoFS",p,r,f))
                        wf.write(eval.format("NoFS",p,r,f)+"\n")
                        y_pred = clf.predict(dbn.predict(dataset.test_data))
                        p = precision(y_pred,dataset.test_label)
                        r = recall(y_pred,dataset.test_label)
                        f = f_measure(y_pred,dataset.test_label)
                        print(eval.format("BFS",p,r,f))
                        wf.write(eval.format("BFS",p,r,f)+"\n")



if __name__ == "__main__":
    np.seterr(all='ignore')
    dataconf = config.dataconf
    DNNconf = config.DNNconf
    datasetConf = dsconf
    projects = ["Xerces","Xalan","Lucene","poi","Synapse"]
    for proj in projects:
        testParameter(dataconf,DNNconf,dsconf,DPtype="WPDP",proj=proj,fixed=["threshold","HiddenLayer","LearningRate","Dimension","CLF"])
