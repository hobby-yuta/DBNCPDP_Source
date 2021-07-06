import sys
import numpy as np

from tqdm import tqdm

from DNN.Layer import Layer
from functions.math import dot,sigmoid,Relu1

def rdc_shape(in_shape,rdc_axis,out_dim):
    out_shape = list(in_shape)
    out_shape[rdc_axis] = out_dim
    return out_shape

def show1rate(x,description='show1rate'):
    print(description,len(x[x == 1])/len(x[x == 0]))


class RBM(Layer):
    def __init__(self,in_shape,rdc_axis,out_dim,lr=0.1,epochs=50,lr_decay=0.95,k=1,rng=None,activation=None):
        #in_shapeは学習データの長さを除いた形 traindata.shape[0:]
        super(RBM, self).__init__(in_shape,rdc_shape(in_shape,rdc_axis,out_dim))
        self.rdc_axis = rdc_axis
        self.lr = lr
        self.epochs = epochs
        self.lr_decay = lr_decay
        self.k = k

        if rng is None:
            #TODO マジックナンバーの消去
            rng = np.random.RandomState(1234)

        if activation is None:
            activation = Relu1

        self.rng = rng
        self.activation = activation

        #重みとバイアスの初期化
        self.reset()

        self.needsPretrain = True
        self.description = 'RBM Layer'

    def getParams(self):
        dict = super(RBM, self).getParams()
        dict["rdc_axis"]:self.rdc_axis
        dict["lr"]:self.lr
        dict["epochs"]:self.epochs
        dict["lr_decay"]:self.lr_decay
        dict["k"]:self.k
        dict["rng"]:self.rng
        dict["activation"]:self.activation
        dict["W"]:self.W
        dict["b"]:self.hb


    def loadParams(self,params):
        super(RBM,self).getParams()
        self.rdc_axis = params["rdc_axis"]
        self.lr = params["lr"]
        self.epochs = params["epochs"]
        self.lr_decay = params["lr_decay"]
        self.k = params["k"]
        self.rng = params["rng"]
        self.activation = params["activation"]
        self.W = params["W"]
        self.b = params["b"]

        self.needsPretrain = False

    def reset(self,W=None,b=None):
        super(RBM, self).reset()

        if W is None:
            a = 1. / self.in_shape[-1]
            W = np.array(self.rng.uniform(
                                low=-a,high=a,
                                size=(self.in_shape[self.rdc_axis],self.out_shape[self.rdc_axis]))
                             )

        if b is None:
            bs = list(self.in_shape)
            bs[self.rdc_axis] = self.out_shape[self.rdc_axis]
            b = np.zeros(bs)

        self.W = W
        self.hb = b
        self.vb = np.zeros(self.in_shape)


    def forward(self,x):
        return self.activation( dot(x,self.W,rdc_axis=self.rdc_axis+1) + self.hb )

    def backward(self,pred,label,early_stop=False):
        print(self.description+" can't back propagation.", file=sys.stderr)
        sys.exit(1)


    def pretrain(self,input,label):
        super(RBM, self).pretrain(input,label)

        h_sample = self.rng.binomial(size=input.shape,n=1,p=input)

        show1rate(h_sample)

        lr = self.lr
        for epoch in tqdm(range(self.epochs)):
            #CD-k 法
            ph_mean, ph_sample = self.sample_h_given_v(h_sample)
            chain_start = ph_sample

            for step in range(self.k):
                if step == 0:
                    nv_means, nv_samples,\
                    nh_means, nh_samples = self.gibbs_hvh(chain_start)
                else:
                    nv_means, nv_samples,\
                    nh_means, nh_samples = self.gibbs_hvh(nh_samples)

            #重みの更新
            if self.rdc_axis != -1 and np.arange(len(h_sample.shape))[-1] != self.rdc_axis+1:
                err = dot(h_sample,ph_sample.transpose(0,2,1)) - dot(nv_samples, nh_means.transpose(0,2,1))
            else:
                err = dot(h_sample.transpose(0,2,1),ph_sample) - dot(nv_samples.transpose(0,2,1), nh_means)
                #print(err.shape)
            self.W += lr * err.mean(axis=0)
            self.vb += lr * np.mean(h_sample - nv_samples, axis=0)
            self.hb += lr * np.mean(ph_sample - nh_means, axis=0)

            lr *= self.lr_decay


    def sample_h_given_v(self,v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.rng.binomial(size=h1_mean.shape,n=1,p=h1_mean)
        return [h1_mean, h1_sample]


    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.rng.binomial(size=v1_mean.shape,n=1,p=v1_mean)
        return [v1_mean, v1_sample]


    def propup(self, v):
        pre_sigmoid_activation = dot(v, self.W,rdc_axis=self.rdc_axis+1) + self.hb

        return self.activation(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = dot(h, self.W.T,rdc_axis=self.rdc_axis+1) + self.vb
        return self.activation(pre_sigmoid_activation)


    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample,
                h1_mean, h1_sample]

    def predict(self, x):
        linear_output = dot(x,self.W,rdc_axis=self.rdc_axis+1) + self.hb
        return self.activation(linear_output)
