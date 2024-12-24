import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from ml_algos.model import BaseModel


class LogisticRegression(BaseModel):
    def __init__(self, num_features, lr=0.1):
        self.lr = lr
        self.W = np.zeros(num_features + 1)
    
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
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, iterations: int):
        for i in range(iterations):
            print(f"iteration {i + 1}/{iterations}: BCE", self.__train(X, y))
    
    def predict(self, X: pd.DataFrame):
        return X.apply(self.__forward, axis="columns")


if __name__ == "__main__":
    X = pd.read_csv("data/pima-indians-diabetes.csv")
    del X["y"]
    y = pd.read_csv("data/pima-indians-diabetes.csv")["y"]

    for col in X.columns:
        X[col] = StandardScaler().fit_transform(X[col].to_numpy().reshape((-1, 1)))

    model = LogisticRegression(8, 0.01)
    model.fit(X, y, 5000)
    print(model.W)
    print(accuracy_score(y, np.where(model.predict(X) > 0.5, 1, 0)))

    plt.scatter(X.index, y)
    plt.scatter(X.index, model.predict(X))
    plt.show()   
