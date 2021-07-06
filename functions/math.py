import sys
import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

def Relu(x):
    return np.maximum(x, 0)

def Relu1(x):
    relu = Relu(x)
    return np.minimum(1,relu)

def step(x):
    return np.where(x > 0, 1, 0)

"""
def dot(a,b):
    if len(a.shape) <= 2:
        return a.dot(b)
    else:
        if a.shape[0] == b.shape[0]:
            return np.array([_a.dot(_b) for _a,_b in zip(a,b)])
        elif len(a.shape) == 3 and len(b.shape) == 2:
            return np.array([_a.dot(b) for _a in a])
"""

def dot(x,W,rdc_axis=-1):
    #print(x.shape,W.shape,rdc_axis)
    if rdc_axis != -1 and np.arange(len(x.shape))[-1] != rdc_axis:
        ts = np.arange(len(x.shape))
        ts[rdc_axis] = ts[-1]
        ts[-1] = rdc_axis
        if len(x.shape) <= 2:
            output = x.transpose(ts).dot(W).transpose(ts)
        elif len(W.shape) == 2:
            ts = ts[1:] -1
            output = np.array([_x.transpose(ts).dot(W).transpose(ts) for _x in x])
        else:
            ts = ts[1:] -1
            output = np.array([_x.transpose(ts).dot(_W).transpose(ts) for _x,_W in zip(x,W)])
    else:
        if len(x.shape) <= 2:
            output = x.dot(W)
        elif len(W.shape) == 2:
            output = np.array([_x.dot(W) for _x in x])
        else:
            output = np.array([_x.dot(_W) for _x,_W in zip(x,W)])
    return output
