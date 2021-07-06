import numpy as np

class sigmoid:
    def forward(self,x):
        return 1. /(1 + np.exp(-x))
    def backward(self,pred,y):
        return y - pred

class Relu1:
    def forward(self,x):
        return np.minimum(1,np.maximum(x,0))
