import numpy as np
import sys

from . import validation

def mean_squared_error(predicted_labels: np.ndarray, 
                       true_labels: np.ndarray) -> float:
    validation.validate_predictions_shape(predicted_labels, true_labels)
    return np.mean(np.square(predicted_labels - true_labels))

def log_loss(predicted_labels: np.ndarray, 
             true_labels: np.ndarray) -> float:
    validation.validate_predictions_shape(predicted_labels, true_labels)
    
    predicted_labels = np.where(predicted_labels > 0, predicted_labels, sys.float_info.min)
    predicted_labels = np.where(predicted_labels < 1, predicted_labels, 1 - sys.float_info.min)
    true_labels = np.where(true_labels < 1, true_labels, 1 - sys.float_info.min)

    return np.sum(-true_labels * np.log(predicted_labels) 
                  - (1 - true_labels) * np.log(1 - predicted_labels))
