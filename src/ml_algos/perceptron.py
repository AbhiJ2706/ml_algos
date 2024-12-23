import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, num_features: int):
        self.W = np.zeros(num_features + 1)

    def __update(self, y: float, x: np.array):
        self.W = self.W + x * y

    def __train_single(self, x: pd.Series):
        y = x["y"]
        y_index = x.index.get_loc("y")
        padded_x = np.concatenate((x.to_numpy()[:y_index], np.ones((1,), dtype=np.float64)))
        _y = np.dot(self.W, padded_x) 
        if y * _y <= 0:
            self.__update(y, padded_x)
        return y * _y
    
    def __train(self, X: pd.DataFrame):
        X.apply(self.__train_single, axis="columns")
    
    def train(self, X: pd.DataFrame, iterations: int):
        for _ in range(iterations):
            self.__train(X)

    def predict_single(self, x: pd.Series):
        y_index = x.index.get_loc("y")
        padded_x = np.concatenate((x.to_numpy()[:y_index], np.ones((1,), dtype=np.float64)))
        return np.dot(self.W, padded_x)
    
    def predict(self, X: pd.DataFrame):
        return X.apply(self.predict_single, axis="columns")
