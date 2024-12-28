import numpy as np
import pandas as pd

from tqdm import trange

from ml_algos.classification_test import iris_test
from ml_algos.model import BaseModel

import math

from enum import Enum


class MultiLayerPerceptron(BaseModel):
    class Activation(Enum):
        SIGMOID = {
            "function": lambda x: 1 / (1 + np.exp(-x)),
            "gradient": lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
        }
        RELU = {
            "function": lambda x: np.max(x, 0),
            "gradient": lambda x: np.max(x, 0) / x
        }
        TANH = {
            "function": lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
            "gradient": lambda x: 1 - (np.exp(x) - np.exp(-x)) ** 2 / (np.exp(x) + np.exp(-x)) ** 2
        }
    
    class Loss(Enum):
        MEAN_SQUARED = {
            "function": lambda x, y: np.mean((x - y) ** 2),
            "gradient": lambda x, y: 2 * (x - y),
        }
    
    class WeightInitialization(Enum):
        GLOROT = lambda n, size: np.random.uniform(-1 / math.sqrt(n), 1 / math.sqrt(n), size=size)
        HE = lambda n, size: np.random.normal(0, 2 / n, size=size)
    
    class GradientDescentMethod(Enum):
        BATCH = 0
        STOCHASTIC = 1
        MINIBATCH = 2

    def __init__(
        self, 
        num_input_features: int, 
        layer_sizes: list[int], 
        activations: list[Activation], 
        loss: Loss,
        weight_initialization: WeightInitialization=WeightInitialization.GLOROT,
        verbose=False
    ):
        self.num_input_features = num_input_features
        self.layer_sizes = layer_sizes
        self.weight_initialization = weight_initialization

        w = [self.num_input_features] + layer_sizes
        self.W = [
            self.weight_initialization(w[i] + 1, (w[i] + 1, w[i + 1])) for i in range(len(w) - 1)
        ]

        self.activations = [func.value["function"] for func in activations]
        self.activation_gradients = [func.value["gradient"] for func in activations]
        self.loss = loss.value["function"]
        self.loss_gradient = loss.value["gradient"]
        self.num_layers = len(self.layer_sizes)

        super(MultiLayerPerceptron, self).__init__(verbose=verbose)
    
    def __forward(self, X: pd.DataFrame):
        X_current = X.to_numpy()
        X_current = np.concatenate((X_current, np.ones((X_current.shape[0], 1))), axis=1)
        self.pre_activations = []
        self.post_activations = [X_current]

        for i in range(self.num_layers):
            self.pre_activations.append(X_current @ self.W[i])
            X_current = np.concatenate((
                self.activations[i](self.pre_activations[-1]), 
                np.ones((self.post_activations[-1].shape[0], 1))), 
            axis=1)
            self.post_activations.append(X_current)

        return self.post_activations[-1][:, :-1]
    
    def __backward(self, H: np.ndarray, y: np.ndarray, learning_rate: float):
        dldH = self.loss_gradient(H, y)
        dldZ = dldH * self.activation_gradients[-1](self.pre_activations[-1])
        weight_gradients = []

        for i in range(1, self.num_layers + 1):
            dldW = (1 / H.shape[0]) * self.post_activations[-i - 1].T @ dldZ
            weight_gradients.append(dldW)
            if i < self.num_layers:
                dldH_1 = dldZ @ self.W[-i].T
                dldH = dldH_1[:, :-1]
                dldZ = dldH * self.activation_gradients[-i - 1](self.pre_activations[-i - 1])
        
        weight_gradients = weight_gradients[::-1]
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - learning_rate * weight_gradients[i]
        return self.loss(H, y)
    
    def __train_batch(self, X: pd.DataFrame, y: np.ndarray, learning_rate: float):
        H = self.__forward(X)
        return self.__backward(H, y, learning_rate)

    def __train_minibatch(self, X: pd.DataFrame, y: np.ndarray, learning_rate: float, batch_size: int):
        num_batches = X.shape[0] // batch_size
        total_loss = 0
        for i in range(num_batches):
            H = self.__forward(X.iloc[i:i + batch_size])
            total_loss += self.__backward(H, y[i:i + batch_size, :], learning_rate)
        return total_loss / num_batches
    
    def __train_stochastic(self, X: pd.DataFrame, y: np.ndarray, learning_rate: float, batch_size: int):
        num_batches = X.shape[0] // batch_size
        max_amount = num_batches * batch_size
        total_loss = 0
        sample_selection = np.random.choice(X.shape[0], max_amount, replace=False)
        X_sampled = X.iloc[sample_selection]
        y_sampled = y[sample_selection, :]
        for i in range(num_batches):
            H = self.__forward(X_sampled[i:i + batch_size])
            total_loss += self.__backward(H, y_sampled[i:i + batch_size, :], learning_rate)
        return total_loss / num_batches

    def _fit(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        iterations: int=100, 
        learning_rate: 
        float=0.1, 
        batch_size: int=32, 
        backward_method: GradientDescentMethod=GradientDescentMethod.BATCH
    ):
        with trange(iterations) as pbar:
            for _ in pbar:
                if backward_method == MultiLayerPerceptron.GradientDescentMethod.BATCH:
                    loss = self.__train_batch(X, y, learning_rate)
                elif backward_method == MultiLayerPerceptron.GradientDescentMethod.MINIBATCH:
                    loss = self.__train_minibatch(X, y, learning_rate, batch_size)
                elif backward_method == MultiLayerPerceptron.GradientDescentMethod.STOCHASTIC:
                    loss = self.__train_stochastic(X, y, learning_rate, batch_size)
                pbar.set_description(f"Loss: {loss:.4f}")
    
    def _predict(self, X: pd.DataFrame):
        return self.__forward(X)


if __name__ == "__main__":
    model = MultiLayerPerceptron(
        4,
        [5, 3],
        [MultiLayerPerceptron.Activation.SIGMOID for _ in range(3)],
        MultiLayerPerceptron.Loss.MEAN_SQUARED,
    )

    iris_test(model, scale=True, one_hot_encode=True, iterations=1000, learning_rate=0.1)
    iris_test(model, scale=True, one_hot_encode=True, iterations=1000, learning_rate=0.1, backward_method=MultiLayerPerceptron.GradientDescentMethod.MINIBATCH)
    iris_test(model, scale=True, one_hot_encode=True, iterations=1000, learning_rate=0.1, backward_method=MultiLayerPerceptron.GradientDescentMethod.STOCHASTIC)
    