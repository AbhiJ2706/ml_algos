import cvxpy as cp
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


class SVMRegressor:
    def __init__(self, epsilon=0.1, c=0.1):
        self._trained = False
        self.epsilon = epsilon
        self.c = c

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame | np.ndarray, prevention: float=1e-3):
        X = X_train.to_numpy()
        y = y_train.to_numpy()

        alpha = cp.Variable(len(X))
        alpha_star = cp.Variable(len(X))
        dot_products = (X @ X.T)

        for i in range(len(X)):
            dot_products[i, i] += prevention

        objective = cp.Maximize(
            -self.epsilon * cp.sum(alpha + alpha_star) +
            cp.sum(cp.multiply(alpha - alpha_star, y)) -
            0.5 * cp.quad_form(alpha - alpha_star, dot_products)
        )
        constraints = [cp.sum(alpha - alpha_star) == 0, alpha >= 0, alpha <= self.c, alpha_star >= 0, alpha_star <= self.c]
        prob = cp.Problem(objective, constraints)

        _ = prob.solve()
        self.W = np.sum([(ai - asi) * xi for ai, asi, xi in zip(alpha.value, alpha_star.value, X)], axis=0)
        self.b = np.mean([yi - np.dot(self.W, xi) for yi, xi in zip(y, X)])
        self._trained = True
    
    def predict(self, X: pd.DataFrame):
        if not self._trained:
            raise ValueError("Model has not been trained")
        
        return X.apply(lambda xi: np.dot(self.W, xi.values) + self.b, axis="columns")


if __name__ == "__main__":
    data = pd.read_csv("data/Salary_dataset.csv")
    for col in data.columns:
        data[col] = StandardScaler().fit_transform(data[col].to_numpy().reshape((-1, 1)))
    model = SVMRegressor(c=1)
    model.fit(pd.DataFrame({"YearsExperience": data["YearsExperience"]}), data["Salary"])
    print(model.W)
    plt.scatter(data["YearsExperience"], model.predict(pd.DataFrame({"YearsExperience": data["YearsExperience"]})))
    plt.scatter(data["YearsExperience"], data["Salary"].to_numpy())
    plt.show()