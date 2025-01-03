import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_algos.model import BaseModel


def diabetes_test(model: BaseModel, scale=False, **training_kwargs):
    data = pd.read_csv("data/pima-indians-diabetes.csv")
    X_train, X_test, y_train, y_test = train_test_split(data, data["y"], test_size=0.33, random_state=42)
    del X_train["y"]
    del X_test["y"]

    if scale:
        for col in X_train.columns:
            X_train[col] = StandardScaler().fit_transform(X_train[col].to_numpy().reshape((-1, 1)))
        for col in X_test.columns:
            X_test[col] = StandardScaler().fit_transform(X_test[col].to_numpy().reshape((-1, 1)))

    model.fit(X_train, y_train, **training_kwargs)
    print(accuracy_score(y_test, np.where(model.predict(X_test) > 0.5, 1, 0)))

    plt.scatter(X_test.index, y_test)
    plt.scatter(X_test.index, model.predict(X_test))
    plt.show()  


def iris_test(model: BaseModel, scale=False, binary=None, one_hot_encode=False, **training_kwargs):
    df = pd.read_csv('data/iris.csv')
    if binary:
        df["variety"] = df["variety"].apply(lambda x: 1 if x == binary else -1)
    elif one_hot_encode:
        df["variety"] = df["variety"].apply(
            lambda x: np.array([1, 0, 0]) if x == "Setosa" else np.array([0, 1, 0]) if x == "Virginica" else np.array([0, 0, 1])
        )

    X_train, X_test, y_train, y_test = train_test_split(df, df["variety"], test_size=0.33, random_state=42)
    del X_train["variety"]
    del X_test["variety"]

    if scale:
        for col in X_train.columns:
            X_train[col] = StandardScaler().fit_transform(X_train[col].to_numpy().reshape((-1, 1)))
        for col in X_test.columns:
            X_test[col] = StandardScaler().fit_transform(X_test[col].to_numpy().reshape((-1, 1)))
    
    if one_hot_encode:
        y_train = np.array(list(map(lambda x: x.tolist(), y_train.to_numpy())))
    
    model.fit(X_train, y_train, **training_kwargs)
    test_result = model.predict(X_test)

    if one_hot_encode:
        y_test = np.array(list(map(lambda x: x.tolist(), y_test.to_numpy())))
        test_result = np.argmax(test_result, axis=1)
        y_test = np.argmax(y_test, axis=1)
        print("accuracy score:", accuracy_score(test_result, y_test))
        print("confusion matrix:\n", confusion_matrix(test_result, y_test))
        return
    
    print("accuracy score:", accuracy_score(test_result.to_numpy(), y_test.to_numpy()))
    print("confusion matrix:\n", confusion_matrix(test_result.to_numpy(), y_test.to_numpy()))
    print("ROC AUC score:", roc_auc_score(test_result.to_numpy(), y_test.to_numpy())) 
    