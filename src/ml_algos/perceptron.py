import numpy as np
import pandas as pd

from ml_algos.classification_test import iris_test
from ml_algos.model import BaseModel


class Perceptron(BaseModel):
    def __init__(self):
        super(BaseModel, self).__init__()

    def __update(self, y: float, x: np.array):
        self.W = self.W + x * y

    def __train_single(self, x: pd.Series, y: int):
        padded_x = np.concatenate((x.to_numpy(), np.ones((1,), dtype=np.float64)))
        _y = np.dot(self.W, padded_x) 
        if y * _y <= 0:
            self.__update(y, padded_x)
    
    def __train(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray):
        for (_, x), y_ in zip(X.iterrows(), y):
            self.__train_single(x, y_)
    
    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray, iterations: int=100):
        self.W = np.zeros(len(X.columns) + 1)
        for _ in range(iterations):
            self.__train(X, y)

    def predict_single(self, x: pd.Series):
        padded_x = np.concatenate((x.to_numpy(), np.ones((1,), dtype=np.float64)))
        return 1 if np.dot(self.W, padded_x) >= 0 else -1
    
    def _predict(self, X: pd.DataFrame):
        return X.apply(self.predict_single, axis="columns")


if __name__ == "__main__":
    iris_test(Perceptron(), binary="Setosa")