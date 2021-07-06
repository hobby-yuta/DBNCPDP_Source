import numpy as np
import random
from functions.math import sigmoid, softmax, Relu1, step, dot


class RBM(object):
    def __init__(self, input=None,n_visible=2, in_dim=2, n_hidden=3, \
        W=None, hbias=None, vbias=None, numpy_rng=None,mask=None,activation=Relu1):

        self.n_visible = n_visible
        self.in_dim = in_dim  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer
        self.activation = activation
        self.mask = mask
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)


        if W is None:
            a = 1. / in_dim
            initial_W = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(in_dim, n_hidden)))

            W = initial_W

        if hbias is None:
            hbias = numpy.zeros(n_visible,n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = numpy.zeros([n_visible, in_dim])  # initialize v bias 0


        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias

        # self.params = [self.W, self.hbias, self.vbias]

    def getParams(self):
        
        return self.n_visible,self.in_dim,self.n_hidden

    def contrastive_divergence(self, lr=0.1, k=1, input=None,mask=None):
        if input is not None:
            self.input = input


        ''' CD-k '''
        ph_mean, ph_sample = self.sample_h_given_v(self.input)
        chain_start = ph_sample

        for step in range(k):
            if step == 0:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(nh_samples)
        # chain_end = nv_samples
        err = (dot(self.input.transpose(0,2,1), ph_sample) - dot(nv_samples.transpose(0,2,1), nh_means))
        self.W += lr * err.mean(axis=0)

        #誤差関数
        self.vbias += lr * numpy.mean(self.input - nv_samples, axis=0)
        self.hbias += lr * numpy.mean(ph_sample - nh_means, axis=0)
        # cost = self.get_reconstruction_cross_entropy()
        # return cost


    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.numpy_rng.binomial(size=h1_mean.shape,n=1,p=h1_mean)
        return [h1_mean, h1_sample]


    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.numpy_rng.binomial(size=v1_mean.shape,n=1,p=v1_mean)
        return [v1_mean, v1_sample]

    def propup(self, v):
        pre_sigmoid_activation = dot(v, self.W) + self.hbias

        return self.activation(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = dot(h, self.W.T) + self.vbias
        return self.activation(pre_sigmoid_activation)


    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample,
                h1_mean, h1_sample]


    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation_h = dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = self.activation(pre_sigmoid_activation_h)

        pre_sigmoid_activation_v = dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = self.activation(pre_sigmoid_activation_v)

        cross_entropy =  - numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation_v) +
            (1 - self.input) * numpy.log(1 - sigmoid_activation_v),
                      axis=1))

        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(dot(h, self.W.T) + self.vbias)
        return reconstructed_v
