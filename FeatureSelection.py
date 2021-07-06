import numpy as np
import copy
from sklearn.model_selection import train_test_split
#from functions.math import dot
from functions.metrics import f_measure,auc
from DNN.Layer import Layer
from DNN.LogisticRegression import LogisticRegression

class FeatureSelectionClassifier(Layer):
    def __init__(self,classifier,input_v,label_v,FS="FFS",\
                epochs=50,k=100,dropout_k=50,dropoutrate=0.0001,metrics=auc):
        self.clf = classifier
        super(FeatureSelectionClassifier, self).__init__(self.clf.in_shape,self.clf.in_shape)
        self.canBackward = True
        self.description = 'Feature Selection Layer'
        self.selected_idx = None

        self.x_v = input_v
        self.y_v = label_v

        self.FS = FS
        self.epochs=epochs
        self.k = k
        self.dropout = dropoutrate
        self.dropout_k = dropout_k
        self.metrics = metrics

        self.reset()

    def reset(self):
        self.needsPretrain = True
        self.clf.reset()

    def forward(self,x):
        if self.selected_idx is None:
            print("Error: feature selection is not completed.", file=sys.stderr)
            sys.exit(1)
        return self.clf.forward(x[:,self.selected_idx])

    def backward(self,x,y,early_stop=False):
        return self.clf.backward(x[:,self.selected_idx],y,early_stop)

    def pretrain(self,input,label):
        super(FeatureSelectionClassifier, self).pretrain(input,label)
        if self.FS == "FFS":
            self.selected_idx = self.FFS(input,label)
        elif self.FS == "BFS":
            self.selected_idx = self.BFS(input,label)
        else:
            print("Error: FS configure isn't defined or wrong.", file=sys.stderr)
            sys.exit(1)
        self.out_shape = [len(self.selected_idx)]
        self.clf.reset(in_dim=len(self.selected_idx))

    def FFS(self,x,y):
        clf = copy.deepcopy(self.clf)
        selected_idx = []
        feature_idx = list(range(x.shape[-1]))
        last_score = 0.
        while len(selected_idx) <= self.k:
            max_score = 0.
            add_idx = np.random.choice(feature_idx,1)[0]
            for f in feature_idx:
                idx = selected_idx + [f]
                clf.reset(in_dim=len(idx))
                for epoch in range(self.epochs):
                    _,early_stop = clf.train(x[:,idx],y)
                    if early_stop:
                        break
                p_v = clf.predict(self.x_v[:,idx])
                score = self.metrics(self.y_v[:,1],p_v[:,1])
                if score > max_score:
                    max_score = score
                    add_idx = f

            selected_idx.append(add_idx)
            feature_idx.remove(add_idx)
            print(add_idx,max_score)
            if len(selected_idx) > self.dropout_k and max_score - last_score < self.dropout:
                print("FFS dropout",len(selected_idx))
                break
            last_score = max_score


        return selected_idx

    def BFS(self,x,y):
        selected_idx = np.arange(x.shape[-1])
        clf = copy.deepcopy(self.clf)
        self.fit(clf,x,y)
        last_score = self.metrics(clf.predict(self.x_v),self.y_v)
        escape_count=0
        escape_tmp = []
        print("base score",last_score)
        while len(selected_idx) > self.k:
            max_score = last_score
            rmv_idx = np.random.choice(selected_idx,1)[0]
            for f in selected_idx:
                idx = selected_idx[selected_idx != f]
                clf.reset(in_dim=len(idx))
                self.fit(clf,x[:,idx],y)
                p_v = clf.predict(self.x_v[:,idx])
                score = self.metrics(p_v,self.y_v)
                if score > max_score:
                    max_score = score
                    rmv_idx = f
                #elif score == max_score:
                #    rmv_idx = np.append(rmv_idx,f)
            #if len(selected_idx) < self.dropout_k and max_score - last_score < self.dropout:
            if max_score - last_score <= 0.0001:
                escape_count += 1
                if escape_count == 3:
                    np.append(selected_idx,np.array(escape_tmp))
                    np.sort(selected_idx)
                    print("BFS dropout",len(selected_idx))
                    break
                escape_tmp.append(rmv_idx)
            else:
                escape_count = 0
                escape_tmp = []
            selected_idx = np.delete(selected_idx,np.where(selected_idx == rmv_idx))
            #print(rmv_idx,max_score)
            #print(selected_idx)
            #last_score += (last_score - max_score) * self.clf.lr
            last_score = max_score
        return selected_idx

    def fit(self,clf,x,y):
        for epoch in range(self.epochs):
            _,early_stop = clf.train(x,y)
            if early_stop:
                break

"""

class BFS:
    def __init__(self,classifier=None,epochs=50,k=100,dropoutrate=0.001,metrics=f_measure,val_size=0.25):
        self.k = k
        self.dropout = dropoutrate
        self.metrics = metrics
        if classifier is None:
            classifier = LogisticRegression(0,2)
        self.clf = classifier
        self.epochs=epochs
        self.val_size=val_size

    def reset_clf(self,classifier):
        self.clf = classifier

    def select(self,x,y,axis=-1):

        selected_idx = np.arange(x.shape[axis])

        x_t, x_v, y_t, y_v = train_test_split(x,y,test_size=self.val_size,shuffle=True,stratify=y)
        self.clf.reset(in_dim=x.shape[axis])
        self.fit(x_t,y_t)
        last_score = self.metrics(self.clf.predict(x_v[:,selected_idx]),y_v)
        print("base score",last_score)
        while len(selected_idx) > self.k:
            #x_t, x_v, y_t, y_v = train_test_split(x,y,test_size=self.val_size,shuffle=True,stratify=y)
            max_score = 0
            rmv_idx = np.random.choice(selected_idx,1)[0]
            for f in selected_idx:
                idx = selected_idx[selected_idx != f]
                self.clf.reset(in_dim=len(idx))
                self.fit(x_t[:,idx],y_t)
                score = self.metrics(self.clf.predict(x_v[:,idx]),y_v)
                if score > max_score:
                    max_score = score
                    rmv_idx = f
                #elif score == max_score:
                #    rmv_idx = np.append(rmv_idx,f)
            if max_score - last_score < self.dropout:
                print("BFS dropout",len(selected_idx))
                break
            print(rmv_idx,max_score,f_measure())
            selected_idx = np.delete(selected_idx,np.where(selected_idx == rmv_idx))
            #last_score += (last_score - max_score) * self.clf.lr
            last_score = max_score
        return selected_idx


    def fit(self,x,y):
        for epoch in range(self.epochs):
            _,early_stop = self.clf.train(x,y)
            if early_stop:
                break


class FFS:
    def __init__(self,classifier=None,epochs=50,k=100,dropout_k=50,dropoutrate=0.001,metrics=f_measure,val_size=0.1):
        self.k = k
        self.dropout = dropoutrate
        self.metrics = metrics
        if classifier is None:
            classifier = LogisticRegression(0,2)
        self.clf = classifier
        self.epochs=epochs
        self.val_size=val_size
        self.dropout_k = dropout_k

    def reset_clf(self,classifier):
        self.clf = classifier

    def select(self,x_t,y_t,x_v,y_v):
        #x_t, x_v, y_t, y_v = train_test_split(x,y,test_size=self.val_size,shuffle=True,stratify=y)
        selected_idx = []
        feature_idx = list(range(x.shape[-1]))
        last_score = 0.
        while len(selected_idx) <= self.k:
            max_score = 0.
            add_idx = np.random.choice(feature_idx,1)[0]
            for f in feature_idx:
                idx = selected_idx + [f]
                self.clf.reset(in_dim=len(idx))
                for epoch in range(self.epochs):
                    _,early_stop = self.clf.train(x_t[:,idx],y_t)
                    if early_stop:
                        break
                p_v = self.clf.predict(x_v[:,idx])
                score = self.metrics(p_v,y_v)
                if score > max_score:
                    max_score = score
                    add_idx = f

            selected_idx.append(add_idx)
            feature_idx.remove(add_idx)
            print("select",add_idx)
            if len(selected_idx) > self.dropout_k and max_score - last_score < self.dropout:
                print("FFS dropout",len(selected_idx))
                break
            last_score = max_score

        return selected_idx

"""
