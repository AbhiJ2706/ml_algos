import numpy as np
import pandas as pd

from tqdm import trange

from ml_algos.classification_test import iris_test
from ml_algos.model import BaseModel
from ml_algos.regression_test import real_estate_test

import math

from enum import Enum


class OptimizerType(Enum):
    DEFAULT = 0
    MOMENTUM = 1
    RMSPROP = 2
    ADAM = 3


class Optimizer:
    def __init__(self, optimizer_type: OptimizerType=OptimizerType.DEFAULT, **kwargs):
        self.optimizer_type = optimizer_type
        if optimizer_type == OptimizerType.MOMENTUM:
            self.momentum_rate = kwargs["momentum_rate"]
            self.optimizer_fn = self.__momentum
        elif optimizer_type == OptimizerType.RMSPROP:
            self.beta = kwargs["beta"]
            self.optimizer_fn = self.__rmsprop
        elif optimizer_type == OptimizerType.ADAM:
            self.beta1 = kwargs["beta1"]
            self.beta2 = kwargs["beta2"]
            self.iteration = 1
            self.optimizer_fn = self.__adam
        elif optimizer_type == OptimizerType.DEFAULT:
            self.optimizer_fn = self.__default
    
    def __call__(self, weight_gradients, learning_rate):
        return self.optimizer_fn(weight_gradients, learning_rate)
    
    def __default(self, weight_gradients, learning_rate):
        return weight_gradients, learning_rate
    
    def __momentum(self, weight_gradients, learning_rate):
        previous_gradients = getattr(self, "previous_gradients", None)
        if previous_gradients:
            new_gradients = [
                gradient + self.momentum_rate * prev for (gradient, prev) in zip(weight_gradients, previous_gradients)
            ]
        else:
            new_gradients = weight_gradients
        self.previous_gradients = new_gradients
        return new_gradients, learning_rate
    
    def __rmsprop(self, weight_gradients, learning_rate):
        previous_gradients = getattr(self, "previous_gradients", None)
        if previous_gradients:
            self.previous_gradients = [
                self.beta * pg + (1 - self.beta) * wg ** 2 for (pg, wg) in zip(self.previous_gradients, weight_gradients)
            ]
        else:
            self.previous_gradients = [
                (1 - self.beta) * wg ** 2 for wg in weight_gradients
            ]
        learning_rates = [
            learning_rate / np.sqrt(pg + 1e-8) for pg in self.previous_gradients
        ]
        weight_gradients = [
            lr * wg for (lr, wg) in zip(learning_rates, weight_gradients)
        ]
        return weight_gradients, 1
    
    def __adam(self, weight_gradients, learning_rate):
        first_moment = getattr(self, "first_moment", None)
        second_moment = getattr(self, "second_moment", None)

        if first_moment is not None and second_moment is not None:
            self.first_moment = [
                self.beta1 * pg + (1 - self.beta1) * wg for (pg, wg) in zip(self.first_moment, weight_gradients)
            ]
            self.second_moment = [
                self.beta2 * pg + (1 - self.beta2) * wg ** 2 for (pg, wg) in zip(self.second_moment, weight_gradients)
            ]
        else:
            self.first_moment = [
                (1 - self.beta1) * wg for wg in weight_gradients
            ]
            self.second_moment = [
                (1 - self.beta2) * wg ** 2 for wg in weight_gradients
            ]
        
        corrected_first_moment = [
            wg / (1 - self.beta1 ** self.iteration) for wg in self.first_moment
        ]
        corrected_second_moment = [
            wg / (1 - self.beta2 ** self.iteration) for wg in self.second_moment
        ]
        
        weight_gradients = [
            learning_rate * fm / (np.sqrt(sm) + 1e-8) for (fm, sm) in zip(corrected_first_moment, corrected_second_moment)
        ]
        self.iteration += 1

        return weight_gradients, 1


class MultiLayerPerceptron(BaseModel):
    class Activation(Enum):
        def _softmax_denominator(x):
            return np.repeat(np.sum(np.exp(x), axis=1), x.shape[1]).reshape(-1, x.shape[1])
        
        SIGMOID = {
            "function": lambda x: 1 / (1 + np.exp(-x)),
            "gradient": lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
        }
        RELU = {
            "function": lambda x: np.where(x >= 0, x, 0),
            "gradient": lambda x: np.where(x >= 0, x, 0) / x
        }
        LINEAR = {
            "function": lambda x: x,
            "gradient": lambda x: 1
        }
        TANH = {
            "function": lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
            "gradient": lambda x: 1 - (np.exp(x) - np.exp(-x)) ** 2 / (np.exp(x) + np.exp(-x)) ** 2
        }
        SOFTMAX = {
            "function": lambda x: np.exp(x) / MultiLayerPerceptron.Activation._softmax_denominator(x),
            "gradient": lambda x: ((np.exp(x) / MultiLayerPerceptron.Activation._softmax_denominator(x)) * 
                                   (1 - np.exp(x) / MultiLayerPerceptron.Activation._softmax_denominator(x)))
        }
    
    class Loss(Enum):
        MEAN_SQUARED = {
            "function": lambda x, y: np.mean((x - y) ** 2),
            "gradient": lambda x, y: 2 * (x - y),
        }
        CROSS_ENTROPY = {
            "function": lambda x, y: -np.sum(y * np.log(x)),
            "gradient": lambda x, y: (x - y),
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
    
    def __backward(self, H: np.ndarray, y: np.ndarray, learning_rate: float, optimizer: Optimizer):
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
        
        weight_gradients, learning_rate = optimizer(weight_gradients[::-1], learning_rate)
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - learning_rate * weight_gradients[i]
        return self.loss(H, y)
    
    def __train_batch(self, X: pd.DataFrame, y: np.ndarray, learning_rate: float):
        H = self.__forward(X)

        return self.__backward(H, y, learning_rate, Optimizer())

    def __train_minibatch(self, X: pd.DataFrame, y: np.ndarray, learning_rate: float, batch_size: int):
        num_batches = X.shape[0] // batch_size
        total_loss = 0

        for i in range(num_batches):
            H = self.__forward(X.iloc[i:i + batch_size])
            total_loss += self.__backward(H, y[i:i + batch_size, :], learning_rate, Optimizer())

        return total_loss / num_batches
    
    def __train_stochastic(self, X: pd.DataFrame, y: np.ndarray, learning_rate: float, batch_size: int, optimizer: Optimizer):
        num_batches = X.shape[0] // batch_size
        max_amount = num_batches * batch_size
        total_loss = 0
        sample_selection = np.random.choice(X.shape[0], max_amount, replace=False)
        X_sampled = X.iloc[sample_selection]
        y_sampled = y[sample_selection, :]

        for i in range(num_batches):
            H = self.__forward(X_sampled[i:i + batch_size])
            total_loss += self.__backward(H, y_sampled[i:i + batch_size, :], learning_rate, optimizer)

        return total_loss / num_batches

    def _fit(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        iterations: int=100, 
        learning_rate: float=0.1, 
        batch_size: int=32, 
        backward_method: GradientDescentMethod=GradientDescentMethod.BATCH,
        optimizer: Optimizer=Optimizer()
    ):
        with trange(iterations) as pbar:
            for _ in pbar:
                if backward_method == MultiLayerPerceptron.GradientDescentMethod.BATCH:
                    loss = self.__train_batch(X, y, learning_rate)
                elif backward_method == MultiLayerPerceptron.GradientDescentMethod.MINIBATCH:
                    loss = self.__train_minibatch(X, y, learning_rate, batch_size)
                elif backward_method == MultiLayerPerceptron.GradientDescentMethod.STOCHASTIC:
                    loss = self.__train_stochastic(X, y, learning_rate, batch_size, optimizer)
                pbar.set_description(f"Loss: {loss:.4f}")
    
    def _predict(self, X: pd.DataFrame):
        return self.__forward(X)


if __name__ == "__main__":
    model = MultiLayerPerceptron(
        4,
        [5, 3],
        [
            MultiLayerPerceptron.Activation.RELU,
            MultiLayerPerceptron.Activation.SOFTMAX
        ],
        MultiLayerPerceptron.Loss.CROSS_ENTROPY,
    )

    iris_test(
        model, 
        scale=True, 
        one_hot_encode=True, 
        iterations=1000, 
        learning_rate=0.1
    )

    model = MultiLayerPerceptron(
        4,
        [5, 3],
        [
            MultiLayerPerceptron.Activation.RELU,
            MultiLayerPerceptron.Activation.SOFTMAX
        ],
        MultiLayerPerceptron.Loss.CROSS_ENTROPY,
    )

    iris_test(
        model, 
        scale=True, 
        one_hot_encode=True, 
        iterations=1000, 
        learning_rate=0.1, 
        backward_method=MultiLayerPerceptron.GradientDescentMethod.MINIBATCH
    )

    model = MultiLayerPerceptron(
        4,
        [5, 3],
        [
            MultiLayerPerceptron.Activation.RELU,
            MultiLayerPerceptron.Activation.SOFTMAX
        ],
        MultiLayerPerceptron.Loss.CROSS_ENTROPY,
    )

    iris_test(
        model, 
        scale=True, 
        one_hot_encode=True, 
        iterations=1000, 
        learning_rate=0.1, 
        backward_method=MultiLayerPerceptron.GradientDescentMethod.STOCHASTIC
    )

    model = MultiLayerPerceptron(
        4,
        [5, 3],
        [
            MultiLayerPerceptron.Activation.RELU,
            MultiLayerPerceptron.Activation.SOFTMAX
        ],
        MultiLayerPerceptron.Loss.CROSS_ENTROPY,
    )

    iris_test(
        model, 
        scale=True, 
        one_hot_encode=True, 
        iterations=100, 
        learning_rate=0.1, 
        backward_method=MultiLayerPerceptron.GradientDescentMethod.STOCHASTIC,
        optimizer=Optimizer(OptimizerType.MOMENTUM, momentum_rate=0.8)
    )

    model = MultiLayerPerceptron(
        4,
        [5, 3],
        [
            MultiLayerPerceptron.Activation.RELU,
            MultiLayerPerceptron.Activation.SOFTMAX
        ],
        MultiLayerPerceptron.Loss.CROSS_ENTROPY,
    )

    iris_test(
        model, 
        scale=True, 
        one_hot_encode=True, 
        iterations=100, 
        learning_rate=0.1, 
        backward_method=MultiLayerPerceptron.GradientDescentMethod.STOCHASTIC,
        optimizer=Optimizer(OptimizerType.RMSPROP, beta=0.9)
    )

    model = MultiLayerPerceptron(
        4,
        [5, 3],
        [
            MultiLayerPerceptron.Activation.RELU,
            MultiLayerPerceptron.Activation.SOFTMAX
        ],
        MultiLayerPerceptron.Loss.CROSS_ENTROPY,
    )

    iris_test(
        model, 
        scale=True, 
        one_hot_encode=True, 
        iterations=75, 
        learning_rate=0.01, 
        backward_method=MultiLayerPerceptron.GradientDescentMethod.STOCHASTIC,
        optimizer=Optimizer(OptimizerType.ADAM, beta1=0.9, beta2=0.999)
    )

    model = MultiLayerPerceptron(
        10,
        [64, 1],
        [
            MultiLayerPerceptron.Activation.RELU,
            MultiLayerPerceptron.Activation.LINEAR
        ],
        MultiLayerPerceptron.Loss.MEAN_SQUARED,
    )

    real_estate_test(
        model, 
        scale=True,
        reshape=True,
        iterations=2500, 
        learning_rate=0.001, 
        backward_method=MultiLayerPerceptron.GradientDescentMethod.BATCH
    )

    model = MultiLayerPerceptron(
        10,
        [64, 1],
        [
            MultiLayerPerceptron.Activation.RELU,
            MultiLayerPerceptron.Activation.LINEAR
        ],
        MultiLayerPerceptron.Loss.MEAN_SQUARED,
    )

    real_estate_test(
        model, 
        scale=True,
        reshape=True,
        iterations=250, 
        learning_rate=0.001, 
        backward_method=MultiLayerPerceptron.GradientDescentMethod.STOCHASTIC,
        optimizer=Optimizer(OptimizerType.MOMENTUM, momentum_rate=0.8)
    )
