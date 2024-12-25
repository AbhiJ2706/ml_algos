import numpy as np
import pandas as pd

from enum import Enum
from math import sqrt

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_algos.model import BaseModel
from ml_algos.regression_test import real_estate_test


class DecisionTree(BaseModel):
    class SearchMethod(Enum):
        BFS = 0
        DFS = -1

    class Node:
        def __init__(self, depth: int=0, min_entries: int=5):
            self.depth: int = depth
            self.min_entries: int = min_entries
            self.leaf_label: float | int = None
            self.left: DecisionTree.Node = None
            self.right: DecisionTree.Node = None
            self.feature: str = None
            self.value: float | int = None
            self.size: int = None

        def fit(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray, force_prediction: bool=False):
            if len(y) <= self.min_entries or force_prediction:
               self.leaf_label = np.mean(y)
               self.size = len(y)
               return None, None
            
            best_feature = None
            best_value = None
            j_loss = float("inf")
            for feat in X.columns:
                thresholds = pd.unique(X[feat])
                for feat_value in thresholds:
                    loss = self.loss(X, y, feat, feat_value)
                    if loss < j_loss:
                        j_loss = loss
                        best_feature = feat
                        best_value = feat_value

            self.feature = best_feature
            self.value = best_value
            return best_feature, best_value
       
        def loss(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray, feat: str, feat_value: float | int):
            lt_mask = X[feat] <= feat_value
            gt_mask = X[feat] > feat_value
            X_lt = X[lt_mask]
            X_gt = X[gt_mask]
            y_lt = y[lt_mask]
            y_gt = y[gt_mask]
            return len(X_lt) * (self.l(y_lt) if len(X_lt) else 0) + len(X_gt) * (self.l(y_gt) if len(X_gt) else 0)
        
        def l(self, y: pd.DataFrame | np.ndarray):
            prediction = np.mean(y)
            return np.mean((y - prediction) ** 2)
        
        def predict(self, X: pd.DataFrame):
            if not self.feature:
                return self.leaf_label
            if X[self.feature] <= self.value:
                if self.left:
                    return self.left.predict(X)
                return self.leaf_label
            elif X[self.feature] > self.value:
                if self.right:
                    return self.right.predict(X)
                return self.leaf_label
            
        def __str__(self):
            if not self.left and not self.right:
                return ' ' * (4 * self.depth) + f"Prediction: {self.leaf_label} ({self.size}, {self.depth})"
            return f"{' ' * (4 * self.depth) if not self.left else ''}{self.left}\n" + \
                f"{' ' * (4 * self.depth)}{self.feature} = {self.value}\n" + \
                f"{' ' * (4 * self.depth) if not self.right else ''}{self.right}"
   
    def __init__(self, max_depth: int=10, min_entries: int=5, search_method: SearchMethod=SearchMethod.BFS):
        self.max_depth = max_depth
        self.min_entries = min_entries
        self.search_method = search_method
        super(DecisionTree, self).__init__()

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray, **kwargs):
        self.head = DecisionTree.Node(min_entries=self.min_entries)
        nodes = [(X, y, self.head)]
        while nodes:
            (X_current, y_current, current_split_node) = nodes.pop(self.search_method.value)
            split_feature, split_value = current_split_node.fit(X_current, y_current, force_prediction=(current_split_node.depth == self.max_depth))
            if split_feature is None:
                continue
            lt_mask = X_current[split_feature] <= split_value
            gt_mask = X_current[split_feature] > split_value
            X_lt = X_current[lt_mask]
            X_gt = X_current[gt_mask]
            y_lt = y_current[lt_mask]
            y_gt = y_current[gt_mask]
            if len(X_lt):
                current_split_node.left = DecisionTree.Node(current_split_node.depth + 1, self.min_entries)
                nodes.append((X_lt, y_lt, current_split_node.left))
            if len(X_gt):
                current_split_node.right = DecisionTree.Node(current_split_node.depth + 1, self.min_entries)
                nodes.append((X_gt, y_gt, current_split_node.right))

    def _predict(self, X: pd.DataFrame):
       return X.apply(self.__predict_single, axis="columns")
    
    def __predict_single(self, x: pd.Series):
       return self.head.predict(x)
    
    def __str__(self):
       return str(self.head)


if __name__ == "__main__":
    real_estate_test(DecisionTree())
