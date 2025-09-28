from typing import Callable

import numpy as np

from .linear_model import LinearModel

import util.metrics
import util.normalization
import util.validation

class LinearRegression(LinearModel):
    def __init__(self, 
                 normalization_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = util.normalization.zscore,
                 loss_func: Callable[[np.ndarray, np.ndarray], float] = util.metrics.mean_squared_error):
        super().__init__(normalization_func, loss_func)

    def __str__(self):
        return 'LinearRegression()'
    
    def __repr__(self):
        return 'LinearRegression()'

    def _update_parameters(self, features:np.ndarray, 
            predicted_labels:np.ndarray, true_labels:np.ndarray) -> None:
        error = predicted_labels - true_labels
        n = features.shape[0]

        bias_derivative = 2 * np.sum(error) / n
        weight_derivatives = 2 * (features.T @ error) / n

        self.bias = self.bias - bias_derivative * self.learning_rate
        self.weights = self.weights - weight_derivatives * self.learning_rate

    def _predict(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights + self.bias
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        util.validation.validate_predict(self.fitted, features, self.weights)

        features = self.normalization_func(features, self.feature_means, self.feature_stdevs)
        return self._predict(features)
