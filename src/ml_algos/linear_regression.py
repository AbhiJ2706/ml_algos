import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


class LinearRegression:
    def __init__(self, num_features, lr=0.1):
        self.lr = lr
        self.W = np.zeros(num_features + 1)

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
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, iterations: int):
        for i in range(iterations):
            print(f"iteration {i + 1}/{iterations}: MSE", self.__train(X, y))
    
    def predict(self, X: pd.DataFrame):
        return X.apply(self.__forward, axis="columns")
    

class RidgeRegression:
    def __init__(self, num_features, lr=0.1, _lambda=0.1):
        self.lr = lr
        self._lambda = _lambda
        self.W = np.zeros(num_features + 1)

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
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, iterations: int):
        for i in range(iterations):
            print(f"iteration {i + 1}/{iterations}: MSE", self.__train(X, y))
    
    def predict(self, X: pd.DataFrame):
        return X.apply(self.__forward, axis="columns")
    

if __name__ == "__main__":
    data = pd.read_csv("data/Salary_dataset.csv")
    model = LinearRegression(1, 0.02)
    model.train(pd.DataFrame({"YearsExperience": data["YearsExperience"]}), data["Salary"], 100)
    print(model.W)
    plt.scatter(data["YearsExperience"], model.W[0] * data["YearsExperience"].to_numpy() + model.W[1])
    plt.scatter(data["YearsExperience"], data["Salary"].to_numpy())
    plt.show()
    