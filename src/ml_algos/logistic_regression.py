import numpy as np
import pandas as pd

from ml_algos.classification_test import diabetes_test
from ml_algos.model import BaseModel


class LogisticRegression(BaseModel):
    def __init__(self, lr=0.1, verbose=False):
        self.lr = lr
        super(LogisticRegression, self).__init__(verbose=verbose)
    
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __backward(self, y, x, X):
        x_arr = np.array(x)
        y_arr = np.array(y)
        loss = np.mean(-y_arr * np.log(np.where(x_arr == 0, 1e-9, x_arr)) - (1 - y_arr) * np.log(np.where(1 - x_arr == 0, 1e-9, 1 - x_arr)))
        bce_grad = (x_arr - y_arr)
        gradients = bce_grad[:, None] * np.concatenate((X.to_numpy(), np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)
        self.W = self.W - self.lr * np.mean(gradients, axis=0)
        return loss

    def __forward(self, x: pd.Series):
        padded_x = np.concatenate((x.to_numpy(), np.ones((1,), dtype=np.float64)))
        return self.__sigmoid(np.dot(self.W, padded_x))
    
    def __train(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray):
        forward = X.apply(self.__forward, axis="columns")
        return self.__backward(y, forward, X)
    
    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, iterations: int=100):
        self.W = np.zeros(len(X.columns) + 1)
        for i in range(iterations):
            loss = self.__train(X, y)
            if self.verbose:
                print(f"iteration {i + 1}/{iterations}: BCE", loss)
    
    def _predict(self, X: pd.DataFrame):
        return X.apply(self.__forward, axis="columns")


if __name__ == "__main__":
    diabetes_test(LogisticRegression(0.01), iterations=5000)
