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


def weakLearn(learn_conf):
    dt = datetime.datetime.now()
    exelog = {"name":"{}_{}_{}_{}{}{}".format(proj,dt.month,dt.day,dt.hour,dt.minute,dt.second)}

    ohvdir = dsconf["WPDP"][proj]["ohv"]
    trainPRJ = dsconf[proj][dsconf["WPDP"][proj]["val"]]
    testPRJ = dsconf[proj][dsconf["WPDP"][proj]["test"]]
    valPRJ = dsconf[proj][dsconf["WPDP"][proj]["train"]]

    dataset = Dataset_WPDP(config.dataconf,ohvdir,trainPRJ,testPRJ,valPRJ)
    input = dataset.train_data
