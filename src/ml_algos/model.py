import numpy as np
import pandas as pd

from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, verbose: bool=False):
        self.verbose = verbose

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray, **kwargs):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs):
        pass
