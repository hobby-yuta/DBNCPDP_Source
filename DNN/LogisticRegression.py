import sys
import numpy as np

from functions.math import dot,sigmoid
from DNN.Layer import Layer

class LogisticRegression(Layer):
    def __init__(self, in_dim, out_dim,\
                lr=0.1,lr_decay=0.95,L2_reg=0.00,\
                earlyStopping=True,dropout=0.01,activation=None,\
                FS=None,rng_seed=123):
        super(LogisticRegression, self).__init__([in_dim],[out_dim])
        self.FS = FS
        self.reset(in_dim, out_dim,lr,lr_decay,L2_reg,earlyStopping,dropout,activation,FS,rng_seed)


        self.selected_idx=None
        self.canBackward = True

    def reset(self, in_dim=None, out_dim=None,\
                lr=None,lr_decay=None,L2_reg=None,\
                earlyStopping=None,dropout=None,activation=None,\
                FS=None,rng_seed=None):
        if in_dim is not None:
            self.in_dim = in_dim
        if out_dim is not None:
            self.out_dim = out_dim
        if lr is not None:
            self.lr = lr
        if not lr_decay is None:
            self.lr_decay = lr_decay
        if L2_reg is not None:
            self.L2_reg = L2_reg
        if earlyStopping is not None:
            self.earlyStopping = earlyStopping
        if dropout is not None:
            self.dropout = dropout
        if activation is not None:
            self.activation = activation
        else:
            self.activation = sigmoid
        if FS is not None:
            self.FS = FS
        if rng_seed is not None:
            self.rng = np.random.RandomState(rng_seed)


        self.W = np.zeros([self.in_dim,self.out_dim])
        self.b = np.zeros(self.out_dim)

        self.__lr = self.lr
        if self.FS is not None:
            self.needsPretrain = True

    def train(self,x,y):
        if self.selected_idx is None:
            return super(LogisticRegression, self).train(x,y)
        else:
            return super(LogisticRegression, self).train(x[:,self.selected_idx],y)

    def forward(self, x):
        return self.activation(dot(x,self.W) + self.b)

    def backward(self,x,y,early_stop=False):
        pred = self.activation(dot(x,self.W) + self.b)
        d_y = y - pred
        err = self.__lr * dot(x.T, d_y) - self.__lr * self.L2_reg * self.W

        self.W += err
        self.b += self.__lr * np.mean(d_y, axis=0)

        self.__lr *= self.lr_decay

        if self.earlyStopping and self.__negative_log_likelihood(x,y) < self.dropout:
            print("Logistic Regression dropouted")
            early_stop = True

        return d_y, early_stop


    def pretrain(self,input,label):
        super(LogisticRegression, self).pretrain(input,label)
        print("input average >",np.mean(input))
        if self.FS is not None:
            print("logistic regression fs start")
            self.FS.reset_clf(LogisticRegression(in_dim=1,out_dim=self.out_dim,\
                        lr=self.lr,lr_decay=self.lr_decay,L2_reg=self.L2_reg,\
                        earlyStopping=False,\
                        activation=self.activation))
            self.selected_idx = self.FS.select(input,label)
            self.reset(in_dim=len(self.selected_idx))
            print("logistic regression fs end",len(self.selected_idx))
    def __negative_log_likelihood(self,x,y):
        sigmoid_activation = self.activation(dot(x, self.W) + self.b)

        cross_entropy = - np.mean(
            np.sum(y * np.log(sigmoid_activation) +
            (1 - y) * np.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


    def predict(self,x):
        if self.selected_idx is None:
            return super(LogisticRegression, self).predict(x)
        else:
            return super(LogisticRegression, self).predict(x[:,self.selected_idx])
