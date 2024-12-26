import numpy as np
import pandas as pd

from ml_algos.classification_test import diabetes_test
from ml_algos.decision_tree_classifier import DecisionTreeClassifier
from ml_algos.decision_tree_regression import DecisionTreeRegressor
from ml_algos.model import BaseModel
from ml_algos.regression_test import real_estate_test

from collections import Counter


class BaggedClassifier(BaseModel):
    def __init__(self, num_models: int, model: BaseModel, **kwargs):
        self.models: list[BaseModel] = [model(**kwargs) for _ in range(num_models)]
        super(BaggedClassifier, self).__init__()
    
    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, random_seed=42, **kwargs):
        np.random.seed(random_seed)
        for model in self.models:
            selection = np.random.choice(X.shape[0], len(X), replace=True)
            X_current = X.iloc[selection]
            y_current = y.iloc[selection]
            model.fit(X_current, y_current)

    def _predict(self, X: pd.DataFrame):
        predictions = np.array([model.predict(X) for model in self.models]).T
        return np.apply_along_axis(lambda x: Counter(x).most_common(n=1)[0][0], 1, predictions)


class BaggedRegressor(BaseModel):
    def __init__(self, num_models: int, model: BaseModel, **kwargs):
        self.models: list[BaseModel] = [model(**kwargs) for _ in range(num_models)]
        super(BaggedRegressor, self).__init__()
    
    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray, random_seed=42, **kwargs):
        np.random.seed(random_seed)
        for model in self.models:
            selection = np.random.choice(X.shape[0], len(X), replace=True)
            X_current = X[selection]
            y_current = y[selection]
            model.fit(X_current, y_current)

    def _predict(self, X: pd.DataFrame):
        predictions = np.array([model.predict(X) for model in self.models]).T
        return np.mean(predictions, axis=1)


class BoostedBinaryClassifier(BaseModel):
    def __init__(self, model: BaseModel, **kwargs):
        self.model: BaseModel = model
        self.model_hyperparameters: dict = kwargs
        super(BoostedBinaryClassifier, self).__init__()
    
    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, boosting_iterations: int=100, random_seed=42, **kwargs):
        self.data_weights = np.array([1 / len(X) for _ in range(X.shape[0])])
        self.final_weights = []
        self.final_models = []
        np.random.seed(random_seed)
        for _ in range(boosting_iterations):
            selection = np.random.choice(X.shape[0], len(X), replace=True, p=self.data_weights)
            current_model: BaseModel = self.model(**self.model_hyperparameters)
            current_model.fit(X.iloc[selection], y.iloc[selection], **kwargs)
            weak_hypothesis = current_model.predict(X)
            accuracy_error = np.sum(self.data_weights * np.where(weak_hypothesis != y, 1, 0))
            alpha_t = 0.5 * np.log((1 - accuracy_error) / accuracy_error)
            self.data_weights = self.data_weights * np.where(weak_hypothesis != y, np.exp(alpha_t), np.exp(-alpha_t))
            self.data_weights = self.data_weights / np.sum(self.data_weights)
            self.final_weights.append(alpha_t)
            self.final_models.append(current_model)
    
    def _predict(self, X: pd.DataFrame):
        return np.where(np.sum([alpha * np.where(model.predict(X) == 0, -1, 1) for (alpha, model) in zip(self.final_weights, self.final_models)], axis=0) >= 0, 1, 0)


if __name__ == "__main__":
    diabetes_test(BoostedBinaryClassifier(DecisionTreeClassifier, max_depth=2, min_entries=1))
    diabetes_test(BaggedClassifier(100, DecisionTreeClassifier, max_depth=5, min_entries=1))
    real_estate_test(BaggedRegressor(100, DecisionTreeRegressor, max_depth=5, min_entries=5))
