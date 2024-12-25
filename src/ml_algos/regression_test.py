import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_algos.model import BaseModel


def salary_test(model: BaseModel, scale=False, **kwargs):
    data = pd.read_csv("data/Salary_dataset.csv")
    if scale:
        for col in data.columns:
            data[col] = StandardScaler().fit_transform(data[col].to_numpy().reshape((-1, 1)))

    model.fit(pd.DataFrame({"YearsExperience": data["YearsExperience"]}), data["Salary"], **kwargs)
    y_test = model.predict(pd.DataFrame({"YearsExperience": data["YearsExperience"]}))

    print(f"R2 score: {r2_score(y_test, data['Salary'])}, MSE: {mean_squared_error(y_test, data['Salary'])}")
    plt.scatter(data["YearsExperience"], model.predict(pd.DataFrame({"YearsExperience": data["YearsExperience"]})))
    plt.scatter(data["YearsExperience"], data["Salary"].to_numpy())
    plt.show()


def real_estate_test(model: BaseModel, scale=False, **kwargs):
    data = pd.read_csv("data/real_estate_dataset.csv")
    if scale:
        for col in data.columns:
            data[col] = StandardScaler().fit_transform(data[col].to_numpy().reshape((-1, 1)))

    X_train, X_test, y_train, y_test = train_test_split(data, data["Price"], test_size=0.33, random_state=42)
    del X_train["ID"]
    del X_test["ID"]
    del X_train["Price"]
    del X_test["Price"]
    
    model.fit(X_train, y_train, **kwargs)

    y_pred = model.predict(X_test)
    print(f"R2 score: {r2_score(y_test, y_pred)}, MSE: {mean_squared_error(y_test, y_pred)}")
