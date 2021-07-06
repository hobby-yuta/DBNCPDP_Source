import numpy as np
import csv
import re
import config as cfg
import random

from tqdm import tqdm


def addBug(label,buggy_rate=0.2):
    bug_size = int(buggy_rate * label.shape[0] - label[:,1].sum())
    normal_idx = np.where(label[:,0] == 1)
    bug_idx = np.random.choice(normal_idx[0],bug_size, replace=False)
    for i in bug_idx:
        label[i][0] = 1
        label[i][1] = 0
    return label

class Dataset_WPDP:
    def __init__(self,dataconf,ohvdir,trainPRJ,testPRJ,valPRJ=None):
        self.dict = []
        self.train_data = []
        self.train_label = []
        self.train_names = []

        self.test_data = []
        self.test_label = []
        self.test_names = []

        if valPRJ is not None:
            self.val_data = []
            self.val_label = []
            self.val_names = []

        outfile = "{}_{}_{}_{}_{}".format(dataconf['threshold'],dataconf['max_len']\
                                        ,dataconf['split_margin'],trainPRJ,testPRJ)
        if valPRJ is not None:
            outfile += "_"+valPRJ

        if os.path.exists(dataconf['path_data_pickle']+outfile):
            with open(dataconf['path_data_pickle']+outfile, 'rb') as f:
                data_clone = pickle.load(f)
                self.train_data = data_clone.train_data
                self.train_label = data_clone.train_label
                self.train_names = data_clone.train_names

                self.test_data = data_clone.test_data
                self.test_label = data_clone.test_label
                self.test_names = data_clone.test_names

                if valPRJ is not None:
                    self.val_data = data_clone.val_data
                    self.val_label = data_clone.val_label
                    self.val_names = data_clone.val_names
        #辞書データの読み込み
        with open('/'.join([dataconf['path_ohv'],ohvdir,'dict.csv']),mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.dict.append(row[1])

        #シーケンスデータの読み込み
        with open('/'.join([dataconf['path_ohv'],ohvdir,'sequences.csv']),mode='r') as f:
            reader = csv.reader(f)
            names = []
            rows = []
            for row in reader:
                #最大長を超えるシーケンスは分割
                if len(row) > dataconf['max_len']+1:
                    for i in range(1,len(row),dataconf['max_len']):
                        begin = i - dataconf['split_margin'] if i != 1 else i
                        names.append(row[0])
                        rows.append([int(r) for r in row[begin:begin+dataconf['max_len']]])
                        #print(len(row[begin:begin+max_len]))
                else:
                    names.append(row[0])
                    rows.append([int(r) for r in row[1:]])

        #後で文字列比較するためにバックスラッシュをスラッシュに変換
        names = [n.replace('\\','/') for n in names]

        #シーケンスリストをnumpyオブジェクトに変える
        data = np.full([len(rows),max([len(r) for r in rows])],-1)
        for i,row in enumerate(rows):
            for j,e in enumerate(row):
                data[i][j] = e




        #data = self.__seqs2oh(data,len(self.dict))
        for i,n in enumerate(names):
            if n.find(trainPRJ) != -1:
                self.train_names.append(n)
                self.train_data.append(data[i])
            elif n.find(testPRJ) != -1:
                self.test_names.append(n)
                self.test_data.append(data[i])
            elif valPRJ is not None and n.find(valPRJ) != -1:
                self.val_names.append(n)
                self.val_data.append(data[i])
            else:
                print('dontmatch')

        with open(dataconf['path_labels']+'/'+trainPRJ+'.csv',mode='r') as f:
            reader = csv.reader(f)
            self.train_label = np.full([len(self.train_names),2],-1)
            for row in reader:
                for i,name in enumerate(self.train_names):
                    if name.find(row[0]) != -1:
                        self.train_label[i] = np.eye(self.train_label.shape[1])[int(float(row[1]))]
            self.train_data,self.train_label = self.__deleteUndefinedData(self.train_data,self.train_label)


        with open(dataconf['path_labels']+'/'+testPRJ+'.csv',mode='r') as f:
            reader = csv.reader(f)
            self.test_label = np.full([len(self.test_names),2],-1)
            for row in reader:
                for i,name in enumerate(self.test_names):
                    if name.find(row[0]) != -1:
                        self.test_label[i] = np.eye(self.test_label.shape[1])[int(float(row[1]))]
            self.test_data,self.test_label = self.__deleteUndefinedData(self.test_data,self.test_label)



        if valPRJ is not None:
            with open(dataconf['path_labels']+'/'+valPRJ+'.csv',mode='r') as f:
                reader = csv.reader(f)
                self.val_label = np.full([len(self.val_names),2],-1)
                for row in reader:
                    for i,name in enumerate(self.val_names):
                        if name.find(row[0]) != -1:
                            self.val_label[i] = np.eye(self.val_label.shape[1])[int(float(row[1]))]
                self.val_data,self.val_label = self.__deleteUndefinedData(self.val_data,self.val_label)

        data = np.concatenate([np.array(self.train_data),np.array(self.test_data),np.array(self.val_data)])
        counts = np.zeros(len(self.dict))
        for seq in data:
            for token in seq:
                counts[token] += 1

        #データのクリーニング
        #thresholdより少ない出現回数のトークンに0を割り当てる
        cleaning = np.where(counts<dataconf['threshold'])[0]
        for i,_ in enumerate(cleaning):
            if cleaning[i] == 0:
                continue
            data = np.where(data == cleaning[i],0,data)
            data = np.where(data > cleaning[i],data-1,data)
            cleaning = np.where(cleaning > cleaning[i],cleaning-1,cleaning)

        for c in cleaning[::-1]:
            if c != 0:
                del self.dict[c]

        idx = len(self.train_data)
        self.train_data = data[:idx]
        self.test_data = data[idx:idx+len(self.test_data)]
        self.val_data = data[idx+len(self.test_data):]

        self.train_data = self.__seqs2oh(self.train_data,len(self.dict))
        self.test_data = self.__seqs2oh(self.test_data,len(self.dict))
        if valPRJ is not None:
            self.val_data = self.__seqs2oh(self.val_data,len(self.dict))

    def __seqs2oh(self,sequences,dictlen):
        data = np.zeros([sequences.shape[0]*sequences.shape[1],dictlen],dtype='int16')
        index = 0
        for seq in tqdm(sequences):
            for token in seq:
                if token != -1:
                    data[index] = np.eye(dictlen)[token]
                    index+=1
                    continue
                index+=1
        return data.reshape(sequences.shape[0],sequences.shape[1],dictlen)

    def __deleteUndefinedData(self,data,labels):
        _data = []
        _labels = []
        for i,label in enumerate(labels):
            if np.all(label != -1):
                _data.append(data[i])
                _labels.append(label)
        return np.array(_data),np.array(_labels)

    #onecoldベクトル ohvectorとあんまり変わらない
    def __seqs2oc(self,sequences,dictlen):
        data = np.full([sequences.shape[0]*sequences.shape[1],dictlen],1,dtype='int16')
        index = 0
        for seq in tqdm(sequences):
            for token in seq:
                if token != -1:
                    data[index] = np.full(dictlen,1) - np.eye(dictlen)[token]
                index+=1
        return data.reshape(sequences.shape[0],sequences.shape[1],dictlen)



if __name__ == '__main__':

    ohvdir = 'log4j1_1To1_2'
    trainPRJ = 'log4j-v_1_1'
    testPRJ = 'log4j-1_2final'

    dataset = Dataset_WPDP(cfg.dataconf,ohvdir,trainPRJ,testPRJ)
    addBug(dataset.train_label)
    #print(dataset.train_data.shape,dataset.test_data.shape)
