import sys
import numpy as np

class Layer:
    def __init__(self,in_shape,out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape
        #基本的に入力は一つ
        #複数入力を与えたい場合は結合層を作成すること
        self.prev_layer = None
        #出力を伝搬する層は何層あっても良いが逆伝搬する層は1つのみにすること
        self.next_layer = None
        self.canBackward = False
        self.description = 'Layer'
        self.needsPretrain = False

    #重みやフラグを初期化する
    def reset(self):
        self.needsPretrain = False

    #変数を辞書型にして保存
    def getParams(self):
        dict = {
            "in_shape":self.in_shape,
            "out_shape":self.out_shape,
            "prev_layer":self.prev_layer,
            "next_layer":self.next_layer,
            "canBackward":self.canBackward,
            "needsPretrain":self.needsPretrain
        }

    def loadParams(self,params):
        self.in_shape = params["in_shape"]
        self.out_shape = params["out_shape"]
        self.prev_layer = params["prev_layer"]
        self.next_layer = params["next_layer"]
        self.canBackward = params["canBackward"]
        self.needsPretrain = params["needsPretrain"]

    #NeruralNetworkのfit時に呼び出される。
    #trainは編集しないでプライベート関数を編集して使う
    def train(self,x,y):
        #if self.needsPretrain:
        #    self.pretrain(x,y)
        output = self.forward(x)
        if self.next_layer is not None:
            given_backward,early_stop = self.next_layer.train(output,y)
        else:
            given_backward = None
            early_stop = None
        if self.canBackward:
            if self.next_layer is not None:
                if given_backward is None:
                    print(self.description+" tried backward but next layer didn't backpropagation.", file=sys.stderr)
                    sys.exit(1)
                #あっているか分からない
                #TODO 動作確認
                backward,early_stop = self.backward(output,given_backward,early_stop)
            else:
                backward,early_stop = self.backward(x,y)
        else:
            backward = given_backward
            early_stop = False if early_stop is None else early_stop
        return backward,early_stop

    #入力xを受け取って順伝搬し、出力yを返す
    def forward(self,x):
        return np.zeros(self.out_shape)

    #出力若しくは次の層の誤差を受け取り逆伝搬をする
    def backward(self,x,y,early_stop=False):
        return np.zeros(self.in_dim),early_stop

    def pretrain(self,input,label):
        self.needsPretrain = False

    def predict(self,x):
        return self.forward(x)


class Dense(Layer):
    def __init__(self,in_shape):
        super(Dense, self).__init__(in_shape,[np.prod(in_shape)])
        self.out_shape = np.prod(in_shape)
    def forward(self,x):
        return x.reshape([x.shape[0],self.out_shape])

    def backward(self,x,y,early_stop=False):
        print(self.description+" can't back propagation.", file=sys.stderr)
        sys.exit(1)
