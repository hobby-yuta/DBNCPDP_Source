import numpy as np
from functions.math import dot,sigmoid

class LogisticRegression(object):
    def __init__(self, in_dim, n_out,init_W=None,init_b=None,\
                lr=0.1,epochs=50,lr_decay=0.95,L2_reg=0.00,\
                earlyStopping=True,dropout=0.01,activation=sigmoid):

        self.reset(in_dim, n_out,init_W,init_b,lr,epochs,lr_decay,L2_reg,earlyStopping,dropout,activation)

    def reset(self, in_dim=None, n_out=None,init_W=None,init_b=None,\
                lr=None,epochs=None,lr_decay=None,L2_reg=None,\
                earlyStopping=None,dropout=None,activation=None):
        if init_W is None:
            self.W = numpy.zeros([in_dim, n_out])  # initialize W 0
        else:
            self.W = W
        if init_b is None:
            self.b = numpy.zeros(n_out)          # initialize bias 0
        else:
            self.b = b

        if not lr is None:
            self.learningrate = lr
        if not epochs is None:
            self.epochs = epochs
        if not lr_decay is None:
            self.lr_decay = lr_decay
        if not L2_reg is None:
            self.L2_reg = L2_reg
        if not earlyStopping is None:
            self.earlyStopping = earlyStopping
        if not dropout is None:
            self.dropout = dropout
        if not activation is None:
            self.activation = activation


    def fit(self,x,y):
        epoch = 0
        lr = self.learningrate
        while epoch < self.epochs:
            p_y_given_x = self.activation(dot(x, self.W) + self.b)
            d_y = y - p_y_given_x

            err = dot(x.T, d_y) - lr * self.L2_reg * self.W

            self.W += lr * err
            self.b += numpy.mean(d_y, axis=0)

            if earlyStopping and self.__negative_log_likelihood(x,y) < self.dropout:
                print("Logistic Regression dropouted,{} epoch".format(epoch))
                break

            epoch += 1
            lr *= self.lr_decay

    def __negative_log_likelihood(self,x,y):
        sigmoid_activation = self.activation(dot(x, self.W) + self.b)

        cross_entropy = - numpy.mean(
            numpy.sum(y * numpy.log(sigmoid_activation) +
            (1 - y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy

    def predict(self, x):
        return self.activation(dot(x, self.W) + self.b)
