from typing import Callable

import numpy as np

from .linear_model import LinearModel

import src.util.metrics
import src.util.normalization
import src.util.validation


class LogisticRegression(LinearModel):
    def __init__(
        self,
        normalization_func: Callable[
            [np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ] = src.util.normalization.zscore,
        loss_func: Callable[
            [np.ndarray, np.ndarray], float
        ] = src.util.metrics.log_loss,
    ):
        super().__init__(normalization_func, loss_func)

    def __str__(self):
        return "LogisticRegression()"

    def __repr__(self):
        return "LogisticRegression()"

    def _update_parameters(
        self,
        features: np.ndarray,
        predicted_labels: np.ndarray,
        true_labels: np.ndarray,
    ) -> None:
        error = predicted_labels - true_labels
        n = features.shape[0]

        bias_derivative = np.sum(error) / n
        weight_derivatives = (features.T @ error) / n

        self.bias = self.bias - bias_derivative * self.learning_rate
        self.weights = self.weights - weight_derivatives * self.learning_rate

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def _predict(self, features: np.ndarray) -> np.ndarray:
        z = features @ self.weights + self.bias
        res = self._sigmoid(z[:])
        return res

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        src.util.validation.validate_predict(self.fitted, features, self.weights)

        features = self.normalization_func(
            features, self.feature_means, self.feature_stdevs
        )
        return self._predict(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        predictions = self.predict_proba(features)
        return (predictions >= 0.5).astype(int)
