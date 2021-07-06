import numpy as np
from sklearn.metrics import roc_auc_score

def accuracy(pred,label):
    return sum(np.argmax(pred,axis=1) == np.argmax(label,axis=1))/len(pred)

def precision(pred,label):
    TP = pred.argmax(axis=1).dot(label.argmax(axis=1))
    FP = pred.argmax(axis=1).dot(np.full(len(pred),1)-label.argmax(axis=1))
    if TP+FP == 0:
        return 0
    return TP/(TP+FP)

def recall(pred,label):
    TP = pred.argmax(axis=1).dot(label.argmax(axis=1))
    FN = (np.full(len(pred),1)-pred.argmax(axis=1)).dot(label.argmax(axis=1))
    if TP+FN == 0:
        return 0
    return TP/(TP+FN)

def f_measure(pred,label):
    TP = pred.argmax(axis=1).dot(label.argmax(axis=1))
    FP = pred.argmax(axis=1).dot(np.full(len(pred),1)-label.argmax(axis=1))
    FN = (np.full(len(pred),1)-pred.argmax(axis=1)).dot(label.argmax(axis=1))
    return TP/(TP+FP/2+FN/2)

def auc(pred,label):
    return roc_auc_score(label,pred)
