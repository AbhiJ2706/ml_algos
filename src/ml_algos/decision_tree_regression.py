import numpy as np
import pandas as pd

from enum import Enum
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_algos.model import BaseModel


class DecisionTree(BaseModel):
    class FeatureChoice(Enum):
        ALL = lambda x: x
        HALF = lambda x: DecisionTree.FeatureChoice.__subsample(x, len(x.columns) // 2)
        ROOT = lambda x: DecisionTree.FeatureChoice.__subsample(x, int(sqrt(len(x.columns))))

        def __subsample(x, amt):
            sampled = np.random.choice(x.columns, size=(amt,), replace=False).tolist()
            if "y" not in sampled:
                sampled += ["y"]
            else:
                sampled += np.random.choice(list(filter(lambda y: y not in sampled, x.columns)), size=(1,), replace=False).tolist() 
            return sampled
        def __str__(x):
            return {
                DecisionTree.FeatureChoice.ALL: "ALL FEATURES",
                DecisionTree.FeatureChoice.HALF: "HALF OF FEATURES",
                DecisionTree.FeatureChoice.ROOT: "ROOT OF NUM FEATURES"
            }[x]

    class Node:
        def __init__(self, X, feature_choice, depth=0):
            self.X = X
            self.y = pd.unique(X["y"])
            self.leaf_label = max(self.y, key=lambda x: len(X[X["y"] == x]))
            self.left = None
            self.right = None
            self.j = None
            self.t = None
            self.depth = depth
            self.feature_choice = feature_choice

        def fit(self, X: pd.DataFrame):
            if len(self.y) <= 5:
               self.leaf_label = np.mean(self.y)
               return None, None
            
            j = None
            t = None
            j_loss = float("inf")
            for feat in self.feature_choice(X):
                if feat == "y":
                    continue
                thresholds = pd.unique(X[feat])
                for feat_value in thresholds:
                    if self.loss(X, feat, feat_value) < j_loss:
                        j_loss = self.loss(X, feat, feat_value)
                        j = feat
                        t = feat_value

            self.j = j
            self.t = t
            return j, t
       
        def loss(self, X, feat, feat_value):
            X_lt = X[X[feat] <= feat_value]
            X_gt = X[X[feat] > feat_value]
            return len(X_lt) * (self.l(X_lt) if len(X_lt) else 0) + len(X_gt) * (self.l(X_gt) if len(X_gt) else 0)
        
        def l(self, X):
            prediction = np.mean(X["y"].to_numpy())
            return np.mean((X["y"].to_numpy() - prediction) ** 2)
        
        def predict(self, X):
            if not self.j:
                return self.leaf_label
            if X[self.j] <= self.t:
                if self.left:
                    return self.left.predict(X)
                return self.leaf_label
            elif X[self.j] > self.t:
                if self.right:
                    return self.right.predict(X)
                return self.leaf_label
            
        def __str__(self):
            if not self.left and not self.right:
                return ' ' * (4 * self.depth) + f"Prediction: {self.leaf_label}"
            return f"{' ' * (4 * self.depth) if not self.left else ''}{self.left}\n" + \
                f"{' ' * (4 * self.depth)}{self.j} = {self.t}\n" + \
                f"{' ' * (4 * self.depth) if not self.right else ''}{self.right}"
   
    def __init__(self, max_depth=10, feature_choice=FeatureChoice.ALL):
        self.max_depth = max_depth
        self.head = None
        self.feature_choice = feature_choice

    def fit(self, X, y):
        training_df = X
        training_df["y"] = y
        self.head = DecisionTree.Node(training_df, self.feature_choice)
        nodes = [(training_df, self.head)]
        while nodes:
            (tdf, current_split_node) = nodes.pop(0)
            if current_split_node.depth == self.max_depth:
                continue
            j, t = current_split_node.fit(tdf)
            if j is None:
                continue
            l_tdf = tdf[tdf[j] <= t]
            r_tdf = tdf[tdf[j] > t]
            if len(l_tdf):
                current_split_node.left = DecisionTree.Node(l_tdf, self.feature_choice, current_split_node.depth + 1)
                nodes.append((l_tdf, current_split_node.left))
            if len(r_tdf):
                current_split_node.right = DecisionTree.Node(r_tdf, self.feature_choice, current_split_node.depth + 1)
                nodes.append((r_tdf, current_split_node.right))

    def predict(self, X):
       return X.apply(self.predict_single, axis="columns")
    
    def predict_single(self, x):
       return self.head.predict(x)
    
    def __str__(self):
       return str(self.head)


if __name__ == "__main__":
    data = pd.read_csv("data/real_estate_dataset.csv")
        
    X_train, X_test, y_train, y_test = train_test_split(data, data["Price"], test_size=0.33, random_state=42)
    del X_train["ID"]
    del X_test["ID"]
    del X_train["Price"]
    del X_test["Price"]
    
    model = DecisionTree()
    model.fit(X_train, y_train)
    print(model)

    y_pred = model.predict(X_test)
    print(f"R2 score: {r2_score(y_test, y_pred)}, MSE: {mean_squared_error(y_test, y_pred)}")
