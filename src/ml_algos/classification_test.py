import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from ml_algos.model import BaseModel


def diabetes_test(model: BaseModel, **kwargs):
    X = pd.read_csv("data/pima-indians-diabetes.csv")
    del X["y"]
    y = pd.read_csv("data/pima-indians-diabetes.csv")["y"]

    for col in X.columns:
        X[col] = StandardScaler().fit_transform(X[col].to_numpy().reshape((-1, 1)))

    model.fit(X, y, **kwargs)
    print(model.W)
    print(accuracy_score(y, np.where(model.predict(X) > 0.5, 1, 0)))

    plt.scatter(X.index, y)
    plt.scatter(X.index, model.predict(X))
    plt.show()   