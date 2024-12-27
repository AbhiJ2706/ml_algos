import numpy as np
import pandas as pd

from ml_algos.model import BaseModel

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MultiLayerPerceptron(BaseModel):
    def __init__(self, num_layers, input_features, layer_sizes, activations, activation_derivatives, loss, loss_gradient, learning_rate):
        self.num_layers = num_layers
        self.input_features = input_features
        self.layer_sizes = layer_sizes
        w = [self.input_features] + layer_sizes
        self.W = [
            np.random.normal(0, 0.05, size=(w[i] + 1, w[i + 1])) for i in range(len(w) - 1)
        ]
        self.activations = activations
        self.activation_derivatives = activation_derivatives
        self.loss = loss
        self.loss_gradient = loss_gradient
        self.learning_rate = learning_rate
        print([w.shape for w in self.W])
        super(MultiLayerPerceptron, self).__init__(verbose=True)
    
    def forward(self, X: pd.DataFrame):
        X_current = X.to_numpy()
        X_current = np.concatenate((X_current, np.ones((X_current.shape[0], 1))), axis=1)
        self.pre_activations = []
        self.post_activations = [X_current]
        for i in range(self.num_layers):
            self.pre_activations.append(X_current @ self.W[i])
            X_current = np.concatenate((self.activations[i](self.pre_activations[-1]), np.ones((self.post_activations[-1].shape[0], 1))), axis=1)
            self.post_activations.append(X_current)
        return self.post_activations[-1][:, :-1]
    
    def backward(self, H: np.ndarray, y: np.ndarray):
        dldH = self.loss_gradient(H, y)
        dldZ = dldH * self.activation_derivatives[-1](self.pre_activations[-1])
        weight_gradients = []
        for i in range(1, self.num_layers + 1):
            dldW = (1 / H.shape[0]) * self.post_activations[-i - 1].T @ dldZ
            weight_gradients.append(dldW)
            if i < self.num_layers:
                dldH_1 = dldZ @ self.W[-i].T
                dldH = dldH_1[:, :-1]
                dldZ = dldH * self.activation_derivatives[-i - 1](self.pre_activations[-i - 1])
        weight_gradients = weight_gradients[::-1]
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - self.learning_rate * weight_gradients[i]
        return self.loss(H, y)
    
    def __train(self, X, y):
        H = self.forward(X)
        return self.backward(H, y)

    def _fit(self, X, y, iterations=100):
        for i in range(iterations):
            loss = self.__train(X, y)
            if self.verbose:
                print(f"iteration {i + 1}/{iterations}: MSE", loss)
    
    def _predict(self, X):
        return self.forward(X)


if __name__ == "__main__":
    df = pd.read_csv('data/iris.csv')
    df["variety"] = df["variety"].apply(
        lambda x: np.array([1, 0, 0]) if x == "Setosa" else np.array([0, 1, 0]) if x == "Virginica" else np.array([0, 0, 1])
    )

    X_train, X_test, y_train, y_test = train_test_split(df, df["variety"], test_size=0.33, random_state=42)
    del X_train["variety"]
    del X_test["variety"]

    for col in X_train.columns:
        X_train[col] = StandardScaler().fit_transform(X_train[col].to_numpy().reshape((-1, 1)))
    for col in X_test.columns:
        X_test[col] = StandardScaler().fit_transform(X_test[col].to_numpy().reshape((-1, 1)))

    model = MultiLayerPerceptron(
        2,
        4,
        [5, 3],
        [lambda x: 1 / (1 + np.exp(-x)) for _ in range(3)],
        [lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x)))) for _ in range(3)],
        lambda x, y: np.mean((x - y) ** 2),
        lambda x, y: 2 * (x - y),
        0.1
    )
    
    model.fit(X_train, np.array(list(map(lambda x: x.tolist(), y_train.to_numpy()))), iterations=1000)
    test_result = model.predict(X_test)

    y_test = np.array(list(map(lambda x: x.tolist(), y_test.to_numpy())))
    
    print("accuracy score:", accuracy_score(np.argmax(test_result, axis=1), np.argmax(y_test, axis=1)))
    print("confusion matrix:\n", confusion_matrix(np.argmax(test_result, axis=1), np.argmax(y_test, axis=1)))
    