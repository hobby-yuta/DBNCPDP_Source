import numpy as np
import random
from functions.math import sigmoid, softmax, Relu1, step, dot


class HiddenLayer(Object):
    def __init__(self, input,layer_size, layer_dim, n_out,\
                 W=None, b=None,\
                 numpy_rng=None,activation=np.tanh):

        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)

        if W is None:
            a = 1. / layer_dim
            initial_W = np.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(layer_dim, n_out)))

            W = initial_W

        if b is None:
            b = np.zeros([layer_size,n_out])  # initialize bias 0


        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.b = b

        self.activation = activation

    def output(self, input=None):
        if input is not None:
            self.input = input
        linear_output = dot(self.input, self.W) + self.b
        return (linear_output if self.activation is None
                else self.activation(linear_output))

    def sample_h_given_v(self, input=None):
        if input is not None:
            self.input = input

        v_mean = self.output()
        h_sample = self.numpy_rng.binomial(size=v_mean.shape,
                                           n=1,
                                           p=v_mean)
        return h_sample


class DimReductionLayer(HiddenLayer):
    def __init__(self, input,layer_dim, n_out,\
                 W=None, b=None,\
                 numpy_rng=None,activation=sigmoid):
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)

        if W is None:
            a = 1. / layer_dim
            initial_W = np.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(layer_dim, n_out)))

            W = initial_W

        if b is None:
            b = np.zeros([layer_size,n_out])  # initialize bias 0

        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.b = b
        self.activation = activation


    def output(self, input=None):
        if input is not None:
            self.input = input
        linear_output = dot(self.input.transpose(0,2,1), self.W) + self.b

class NodeReductionLayer(HiddenLayer):
    def __init__(self, input,node_size, n_out,\
                 W=None, b=None,\
                 numpy_rng=None,activation=sigmoid):
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)

        if W is None:
            a = 1. / layer_dim
            initial_W = np.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(node_size, n_out)))

            W = initial_W

        if b is None:
            b = np.zeros([n_out,input.shape[-1]])  # initialize bias 0

        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.b = b
        self.activation = activation


    def output(self, input=None):
        if input is not None:
            self.input = input
        linear_output = dot(self.input.transpose(0,2,1), self.W).transpose(0,2,1) + self.b
