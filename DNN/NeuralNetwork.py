import sys

import numpy as np

from DNN.Layer import Layer

class NeuralNetwork:
    def __init__(self,lr=0.1,epochs=50):
        self.lr = lr
        self.epochs = 50
        self.first_layer = None
        self.last_layer = None

    #pretrainもfinetuneも一気にやる
    def fit(self,x,y):
        if self.first_layer is None:
            print("Error : NN doesn't contain any layer.", file=sys.stderr)
            sys.exit(1)

        layer = self.first_layer
        input = x
        while layer is not None:
            if layer.needsPretrain:
                layer.pretrain(input,y)
            input = layer.predict(input)
            layer = layer.next_layer



        epoch = 0
        lr = self.lr
        while epoch < self.epochs:
            _,early_stop = self.first_layer.train(x,y)
            if early_stop:
                break

            epoch += 1

    def predict(self,x):
        if self.first_layer is None:
            print('Error: needs define first layer',file=sys.stderr)
            exit(1)
        output = self.first_layer.predict(x)
        layer = self.first_layer
        while layer.next_layer is not None:
            layer = layer.next_layer
            output = layer.predict(output)
        return output

    def add(self, layer):
        if self.first_layer is None:
            self.first_layer = layer
            self.last_layer = layer
        else:
            self.last_layer.next_layer = layer
            self.last_layer = layer

    def saveDNN(self):
        
