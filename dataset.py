import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



def loadIris(binary=False,test_split=0.1):
    iris = load_iris()
    data = iris.data
    target = iris.target

    if binary:
        mean = np.zeros(len(iris.data[0]))
        np.mean(data,axis=0,out=mean)
        mean = np.array([mean.tolist()] * len(iris.data))
        data = np.where(data > mean, 1,0)
        target = np.eye(3)[target]

    return train_test_split(data,target,test_size=test_split,stratify=target)

def loadIris_quatile(test_split=0.1):
    iris = load_iris()
    data = iris.data
    target = iris.target
    q25,q50,q75 = np.percentile(data,[25,50,75],axis=0)
    q25 = np.array([q25.tolist()] * data.shape[0])
    q50 = np.array([q50.tolist()] * data.shape[0])
    q75 = np.array([q75.tolist()] * data.shape[0])
    data_new = np.zeros(data.shape,dtype=np.int16)
    data_new = np.where(data > q25, data_new + 1, data_new)
    data_new = np.where(data > q50, data_new + 1, data_new)
    data_new = np.where(data > q75, data_new + 1, data_new)
    data = np.array([np.concatenate([np.eye(4)[dn[0]],np.eye(4)[dn[1]],np.eye(4)[dn[2]],np.eye(4)[dn[3]]],axis=0) for dn in data_new])

    target = np.eye(3)[target]

    print(data)

    return train_test_split(data,target,test_size=test_split,stratify=target)


if __name__ == "__main__":
    loadIris_quatile()
