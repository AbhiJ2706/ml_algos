import numpy as np
import pandas as pd

from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, verbose: bool=False):
        self.verbose = verbose
        self._trained = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray, **kwargs):
        self._fit(X, y, **kwargs)
        self._trained = True
    
    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray, **kwargs):
        pass

    def predict(self, X: pd.DataFrame, **kwargs):
        if not self._trained:
            raise ValueError("Model has not been trained")
        
        return self._predict(X, **kwargs)
    
    @abstractmethod
    def _predict(self, X: pd.DataFrame, **kwargs):
        pass
