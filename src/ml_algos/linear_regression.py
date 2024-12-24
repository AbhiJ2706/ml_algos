import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_algos.model import BaseModel


class LinearRegression(BaseModel):
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
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray, iterations: int):
        for i in range(iterations):
            print(f"iteration {i + 1}/{iterations}: MSE", self.__train(X, y))
    
    def predict(self, X: pd.DataFrame):
        return X.apply(self.__forward, axis="columns")
    

class RidgeRegression(BaseModel):
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
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, iterations: int):
        for i in range(iterations):
            print(f"iteration {i + 1}/{iterations}: MSE", self.__train(X, y))
    
    def predict(self, X: pd.DataFrame):
        return X.apply(self.__forward, axis="columns")
    

if __name__ == "__main__":
    data = pd.read_csv("data/Salary_dataset.csv")
    model = LinearRegression(1, 0.02)
    model.fit(pd.DataFrame({"YearsExperience": data["YearsExperience"]}), data["Salary"], 100)
    print(model.W)

    y_test = model.predict(pd.DataFrame({"YearsExperience": data["YearsExperience"]}))
    print(f"R2 score: {r2_score(y_test, data['Salary'])}, MSE: {mean_squared_error(y_test, data['Salary'])}")
    plt.scatter(data["YearsExperience"], y_test)
    plt.scatter(data["YearsExperience"], data["Salary"].to_numpy())
    plt.show()

    data = pd.read_csv("data/Salary_dataset.csv")
    model = RidgeRegression(1, 0.02)
    model.fit(pd.DataFrame({"YearsExperience": data["YearsExperience"]}), data["Salary"], 100)
    print(model.W)

    y_test = model.predict(pd.DataFrame({"YearsExperience": data["YearsExperience"]}))
    print(f"R2 score: {r2_score(y_test, data['Salary'])}, MSE: {mean_squared_error(y_test, data['Salary'])}")
    plt.scatter(data["YearsExperience"], y_test)
    plt.scatter(data["YearsExperience"], data["Salary"].to_numpy())
    plt.show()

    data = pd.read_csv("data/real_estate_dataset.csv")
    for col in data.columns:
        data[col] = StandardScaler().fit_transform(data[col].to_numpy().reshape((-1, 1)))

    X_train, X_test, y_train, y_test = train_test_split(data, data["Price"], test_size=0.33, random_state=42)
    del X_train["ID"]
    del X_test["ID"]
    del X_train["Price"]
    del X_test["Price"]
    
    model = LinearRegression(len(X_train.columns), 0.01)
    model.fit(X_train, y_train, 100)
    print(model.W)

    y_pred = model.predict(X_test)
    print(f"R2 score: {r2_score(y_test, y_pred)}, MSE: {mean_squared_error(y_test, y_pred)}")
    