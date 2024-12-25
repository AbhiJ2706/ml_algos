import cvxpy as cp
import numpy as np
import pandas as pd

from sklearn.gaussian_process.kernels import RBF

from ml_algos.classification_test import iris_test
from ml_algos.model import BaseModel

from typing import Any


class HardMarginSVMClassifier(BaseModel):
    def __init__(self):
        pass

    def _fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame | np.ndarray, epsilon: float=1e-3):
        X = X_train.to_numpy()
        y = y_train.to_numpy()

        lambda_ = cp.Variable(len(X))
        dot_products = (X @ X.T)

        for i in range(len(X)):
            dot_products[i, i] += epsilon

        objective = cp.Maximize(cp.sum(lambda_) - 0.5 * cp.quad_form(cp.multiply(lambda_, y), dot_products))
        constraints = [lambda_ >= 0, cp.sum(cp.multiply(lambda_, y)) == 0]
        prob = cp.Problem(objective, constraints)

        _ = prob.solve()
        self.W = np.sum([li * yi * xi for li, yi, xi in zip(lambda_.value, y, X)], axis=0)
        self.b = np.mean([yi - np.dot(self.W, xi) for yi, xi in zip(y, X)])
    
    def _predict(self, X: pd.DataFrame):
        return X.apply(lambda xi: 1 if np.dot(self.W, xi.values) + self.b >= 0 else -1, axis="columns")


class SoftMarginSVMClassifier(BaseModel):
    def __init__(self, c: float):
        self.c = c

    def _fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame | np.ndarray, epsilon: float=1e-3):
        X = X_train.to_numpy()
        y = y_train.to_numpy()

        lambda_ = cp.Variable(len(X))
        dot_products = (X @ X.T)

        for i in range(len(X)):
            dot_products[i, i] += epsilon

        objective = cp.Maximize(cp.sum(lambda_) - 0.5 * cp.quad_form(cp.multiply(lambda_, y), dot_products))
        constraints = [lambda_ >= 0, lambda_ <= self.c, cp.sum(cp.multiply(lambda_, y)) == 0]
        prob = cp.Problem(objective, constraints)

        _ = prob.solve()
        self.W = np.sum([li * yi * xi for li, yi, xi in zip(lambda_.value, y, X)], axis=0)
        self.b = np.mean([yi - np.dot(self.W, xi) for yi, xi in zip(y, X)])
    
    def _predict(self, X: pd.DataFrame):
        return X.apply(lambda xi: 1 if np.dot(self.W, xi.values) + self.b >= 0 else -1, axis="columns")


class KernelSVMClassifier(BaseModel):
    def __init__(self, c: float, kernel: Any=RBF()):
        self.c = c
        self.kernel = kernel

    def _fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame | np.ndarray, epsilon: float=1e-3):
        X = X_train.to_numpy()
        y = y_train.to_numpy()

        lambda_ = cp.Variable(len(X))
        dot_products = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                dot_products[i, j] = np.sum(self.kernel(X[i, :].reshape((-1, 1)), X[j, :].reshape((-1, 1))))
        
        for i in range(len(X)):
            dot_products[i, i] += epsilon

        objective = cp.Maximize(cp.sum(lambda_) - 0.5 * cp.quad_form(cp.multiply(lambda_, y), dot_products))
        constraints = [lambda_ >= 0, cp.sum(cp.multiply(lambda_, y)) >= 0, lambda_ <= self.c]
        prob = cp.Problem(objective, constraints)

        _ = prob.solve()
        self.W = np.sum([li * yi * xi for li, yi, xi in zip(lambda_.value, y, X)], axis=0)
        self.b = np.mean([yi - np.dot(self.W, xi) for yi, xi in zip(y, X)])
        self._trained = True
    
    def _predict(self, X: pd.DataFrame):
        return X.apply(lambda xi: 1 if np.dot(self.W, xi.values) + self.b >= 0 else -1, axis="columns")
    

if __name__ == "__main__":
    iris_test(HardMarginSVMClassifier(), scale=True, binary="Setosa")
    iris_test(SoftMarginSVMClassifier(0.01), scale=True, binary="Virginica")
    iris_test(KernelSVMClassifier(0.01), scale=True, binary="Virginica")
