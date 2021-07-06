#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Deep Belief Nets (DBN)
 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials
'''

import sys
import numpy
import random
#繰り返しの進捗表示
from tqdm import tqdm
from makedata import Dataset_WPDP,addBug
from functions.math import sigmoid, softmax, Relu1, step, dot
from functions.metrics import accuracy, precision, recall, f_measure
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
import time
import config
import RandomForest
from LogisticRegression import LogisticRegression

numpy.seterr(all='ignore')

#全結合
def dense(x):
    return x.reshape([x.shape[0],x.shape[1]*x.shape[2]])

#入力onehotデータ内の1の割合を調べる
#0が表示された場合何の意味もないデータ
def show1rate(x,description='show1rate'):
    print(description,len(x[x == 1])/len(x[x == 0]))

def masking_oh(x):
    masking=x.sum(axis=-1)
    hotrate = numpy.tile(numpy.full(masking.shape[1],1)/masking.sum(axis=0),[masking.shape[1],1])
    return hotrate

def maskforward(err,mask):
    print(err.shape,mask.shape)
    error = err.sum(axis=0)
    for i,e in enumerate(error):
        e /= mask.sum(axis=1)[i]
    return error

class DBN(object):
    def __init__(self, input=None, label=None,\
                 n_visible=10,in_dim=2, hidden_layer_sizes=[3, 3], n_outs=2,\
                 numpy_rng=None,log_layer=None):

        self.x = input
        self.y = label

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)  # = len(self.rbm_layers)
        self.cost = -1

        #隠れ層の活性化関数 Relu1推奨
        self.hidden_activation = Relu1

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)


        assert self.n_layers > 0


        # construct multi-layer
        for i in range(self.n_layers):
            # layer_size
            if i == 0:
                input_size = in_dim
            else:
                input_size = hidden_layer_sizes[i - 1]

            # layer_input
            if i == 0:
                layer_input = self.x
                mask = masking_oh(self.x)
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()
                mask = None
            # construct sigmoid_layer
            sigmoid_layer = HiddenLayer(input=layer_input,
                                        n_visible=n_visible,
                                        in_dim=input_size,
                                        n_out=hidden_layer_sizes[i],
                                        numpy_rng=numpy_rng,
                                        activation=self.hidden_activation)
            self.sigmoid_layers.append(sigmoid_layer)

            # construct rbm_layer
            rbm_layer = RBM(input=layer_input,
                            n_visible = n_visible,
                            in_dim=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            W=sigmoid_layer.W,     # W, b are shared
                            hbias=sigmoid_layer.b,
                            mask=mask,
                            activation=self.hidden_activation)
            self.rbm_layers.append(rbm_layer)


        # layer for output using Logistic Regression
        if log_layer is None:
            self.log_layer = LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                                label=self.y,
                                                n_visible=n_visible,
                                                in_dim=hidden_layer_sizes[-1],
                                                n_out=n_outs)
        else:
            self.log_layer = log_layer
            log_layer.reset(input = self.sigmoid_layers[-1].sample_h_given_v(),
                            label=self.y)

        # finetune cost: the negative log likelihood of the logistic regression layer
        self.finetune_cost = -1


    def pretrain(self, lr=0.1, k=1, epochs=100,dropout=0.0001):
        # pre-train layer-wise
        print("pretrain start")

        for i in range(self.n_layers):
            learningrate = lr
            print("layer {:2d}".format(i))
            self.cost = -1
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i-1].sample_h_given_v(layer_input)

            show1rate(layer_input,"layer-{} 1rate".format(i))
            rbm = self.rbm_layers[i]

            for epoch in tqdm(range(epochs)):
                if i == 0:
                    #入力のzeroパディングをマスキングする(未実装)
                    #mask = layer_input.max(axis=2).sum(axis=1)
                    rbm.contrastive_divergence(lr=learningrate, k=k, input=layer_input)
                else:
                    rbm.contrastive_divergence(lr=learningrate, k=k, input=layer_input)

            #ドロップアウトする気配がないので高速化のためコメントアウト
                """
                if self.cost == -1:
                    self.cost = rbm.get_reconstruction_cross_entropy()
                else:
                    cost = rbm.get_reconstruction_cross_entropy()
                    #print(cost)
                    if abs(self.cost - cost) < dropout:
                        print("pretrain dropouted, {}".format(epoch))
                        #print(rbm.W)
                        self.cost = -1
                        break
                    self.cost = cost
                """
                #学習率の変化率
                #学習の様子を見て変化させる
                #0.9～1.0ぐらい?
                learningrate *= 0.95

    def finetune(self, lr=0.1, epochs=100,dropout=0.01):
        layer_input = self.x
        for i in range(self.n_layers):
            layer_input = self.sigmoid_layers[i].output(layer_input)

        #分類器への入力の表示
        #print(layer_input)

        # train log_layer
        epoch = 0

        print("finetune start")
        #show1rate(layer_input,"loglayer 1rate")
        for epoch in range(epochs):
            self.log_layer.train(lr=lr, input=layer_input)

            cost = self.log_layer.negative_log_likelihood()
            if self.finetune_cost == -1:
                self.finetune_cost = cost
            else:
                if abs(self.finetune_cost - cost) < dropout:
                    print("finetune dropouted, {}".format(epoch))
                    break
                self.finetune_cost = cost

            lr *= 0.95


    def predict(self, x):
        layer_input = x

        for i in range(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            # rbm_layer = self.rbm_layers[i]
            layer_input = sigmoid_layer.output(input=layer_input)
        out = self.log_layer.predict(layer_input)
        return out

    def output(self,x=None,iter=None):
        if x is None:
            x = self.x
        if iter is None:
            iter = self.n_layers
        for i in range(iter):
            x = self.sigmoid_layers[i].output(x)
        return x

class HiddenLayer(object):
    def __init__(self, input,n_visible, in_dim, n_out,\
                 W=None, b=None,\
                 numpy_rng=None,activation=numpy.tanh):

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if W is None:
            a = 1. / in_dim
            initial_W = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(in_dim, n_out)))

            W = initial_W

        if b is None:
            b = numpy.zeros([n_visible,n_out])  # initialize bias 0


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


    def negative_log_likelihood(self):
        sigmoid_activation = self.activation(dot(self.x, self.W) + self.b)

        cross_entropy = - numpy.mean(
            numpy.sum(self.y * numpy.log(sigmoid_activation) +
            (1 - self.y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


def test_dbn(dataset,pretrain_lr=0.01, pretraining_epochs=100, k=1, \
             finetune_lr=0.01, finetune_epochs=200,layer = [100]):

    hidden_layer_sizes = layer

    #学習用ラベル
    #addBugでbuggy_rateまでバグを混入させる
    trainlabel = dataset.train_label
    #print('ansrate',ansrate(dataset.train_label),ansrate(dataset.test_label))
    rng = numpy.random.RandomState(123)
    # construct DBN
    dbn = DBN(input=dataset.train_data, label=trainlabel,n_visible=dataset.train_data.shape[1], in_dim=dataset.train_data.shape[-1], hidden_layer_sizes=hidden_layer_sizes, n_outs=dataset.train_label.shape[1], numpy_rng=rng)
    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs,dropout=0.01)

    rf = RandomForest(tree_size=100)
    rf.train(dense(dbn.output()),trainlabel)

    # fine-tuning (DBNSupervisedFineTuning)
    #dbn.finetune(lr=finetune_lr, epochs=finetune_epochs,dropout=0.01)

    # test
    #y_pred = dbn.predict(dataset.test_data)
    y_pred = rf.predict(dense(dbn.output(dataset.test_data)))

    #予測結果とテストラベルの比較
#    for i,_ in enumerate(y_pred[:10]):
#        print(y_pred[i],dataset.test_label[i])


    print('accuracy',accuracy(y_pred,dataset.test_label))
    p = precision(y_pred,dataset.test_label)
    print('precision',p)
    r = recall(y_pred,dataset.test_label)
    print('recall',r)
    print('f-measure',2*p*r/(p+r))

    return dbn

def ansrate(pred):
    rate = numpy.zeros(pred.shape[1])
    for idx in pred.argmax(axis=1):
        rate[idx] += 1
    return rate/pred.shape[0]



if __name__ == "__main__":
    #dict.csvとsequence.csvを入れるディレクトリ名
    #githubohvの下に置く
    ohvdir = 'Xerces-J_1_2To1_3'

    #訓練用のjavaプロジェクトディレクトリの名前
    #labels下のcsvファイルもこの名前にしておく
    #ソースコードをパースする時にこの名前にしておく
    trainPRJ = 'Xerces-J_1_2_0'
    #テスト用のjavaプロジェクトディレクトリの名前
    #↑と同様
    testPRJ = 'Xerces-J_1_3_0'

    dataset = Dataset_WPDP(config.dataconf,ohvdir,trainPRJ,testPRJ)

    #numpy行列の表示数の上限を無くす、3次元行列などを表示するときはスライスを使わないとすごいことになる
    #numpy.set_printoptions(threshold=numpy.inf)

    num = 5
    layer = [100] * num
    dbn = test_dbn(dataset,pretrain_lr=0.15,finetune_lr=0.15,pretraining_epochs=50,finetune_epochs=100,layer=layer)

    for i in range(1,num+1):
        print('\nlayer num',i)
        rf = RandomForest(tree_size=100)
        rf.train(dense(dbn.output(iter=i)),dataset.train_label)
        y_pred = rf.predict(dense(dbn.output(dataset.test_data,iter=i)))
        print('accuracy',accuracy(y_pred,dataset.test_label))
        p = precision(y_pred,dataset.test_label)
        print('precision',p)
        r = recall(y_pred,dataset.test_label)
        print('recall',r)
        print('f-measure',2*p*r/(p+r))
