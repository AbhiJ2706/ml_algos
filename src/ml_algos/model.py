import numpy as np
import pandas as pd

from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray, **kwargs):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs):
        pass
