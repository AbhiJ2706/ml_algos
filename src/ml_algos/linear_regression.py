import numpy as np
import pandas as pd

from ml_algos.model import BaseModel
from ml_algos.regression_test import real_estate_test, salary_test


class LinearRegression(BaseModel):
    def __init__(self, lr=0.1, verbose=False):
        self.lr = lr
        super(LinearRegression, self).__init__(verbose=verbose)

    def __backward(self, y: pd.Series, x: pd.Series, X: pd.DataFrame):
        x_arr = np.array(x)
        y_arr = np.array(y)
        loss = np.mean((x_arr - y_arr) ** 2)
        mse_grad = 2 * (x_arr - y_arr)
        gradients = mse_grad[:, None] * np.concatenate((X.to_numpy(), np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)
        self.W = self.W - self.lr * np.mean(gradients, axis=0)
        return loss

    def __forward(self, x: pd.Series):
        padded_x = np.concatenate((x.to_numpy(), np.ones((1,), dtype=np.float64)))
        return np.dot(self.W, padded_x)
    
    def __train(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray):
        forward = X.apply(self.__forward, axis="columns")
        return self.__backward(y, forward, X)
    
    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray, iterations: int):
        self.W = np.zeros(len(X.columns) + 1)
        for i in range(iterations):
            loss = self.__train(X, y)
            if self.verbose:
                print(f"iteration {i + 1}/{iterations}: MSE", loss)
    
    def _predict(self, X: pd.DataFrame):
        return X.apply(self.__forward, axis="columns")
    

class RidgeRegression(BaseModel):
    def __init__(self, lr=0.1, _lambda=0.1, verbose=False):
        self.lr = lr
        self._lambda = _lambda
        super(RidgeRegression, self).__init__(verbose=verbose)

    def __backward(self, y: pd.Series, x: pd.Series, X: pd.DataFrame):
        x_arr = np.array(x)
        y_arr = np.array(y)
        loss = np.mean((x_arr - y_arr) ** 2) + self._lambda * np.mean(self.W ** 2)
        mse_grad = 2 * (x_arr - y_arr)
        gradients = mse_grad[:, None] * np.concatenate((X.to_numpy(), np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)
        self.W = self.W - self.lr * (np.mean(gradients, axis=0) + self._lambda / X.shape[0] * self.W)
        return loss

    def __forward(self, x: pd.Series):
        padded_x = np.concatenate((x.to_numpy(), np.ones((1,), dtype=np.float64)))
        return np.dot(self.W, padded_x)
    
    def __train(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray):
        forward = X.apply(self.__forward, axis="columns")
        return self.__backward(y, forward, X)
    
    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, iterations: int):
        self.W = np.zeros(len(X.columns) + 1)
        for i in range(iterations):
            loss = self.__train(X, y)
            if self.verbose:
                print(f"iteration {i + 1}/{iterations}: MSE", loss)
    
    def _predict(self, X: pd.DataFrame):
        return X.apply(self.__forward, axis="columns")
    

if __name__ == "__main__":
    salary_test(LinearRegression(0.02), iterations=100)
    salary_test(RidgeRegression(0.02), iterations=100)

    real_estate_test(LinearRegression(0.01), iterations=100, scale=True)
    real_estate_test(RidgeRegression(0.01), iterations=100, scale=True)
    