import numpy as np
from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self,tree_size=10):
        self.clf = RandomForestClassifier(n_estimators=tree_size,criterion="gini",min_impurity_decrease=0.0005 ,class_weight="balanced_subsample")

    def train(self,x,y):
        self.clf.fit(x,y)

    def predict(self,x):
        return self.clf.predict(x)
