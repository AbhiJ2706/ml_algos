import cvxpy as cp
import numpy as np
import pandas as pd

from ml_algos.model import BaseModel
from ml_algos.regression_test import real_estate_test, salary_test


class SVMRegressor(BaseModel):
    def __init__(self, epsilon: float=0.1, c: float=0.1):
        self.epsilon = epsilon
        self.c = c
        super(SVMRegressor, self).__init__()

    def _fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame | np.ndarray, prevention: float=1e-3):
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
    
    def _predict(self, X: pd.DataFrame):
        return X.apply(lambda xi: np.dot(self.W, xi.values) + self.b, axis="columns")


if __name__ == "__main__":
    salary_test(SVMRegressor(c=1), scale=True)
    real_estate_test(SVMRegressor(c=1), scale=True)