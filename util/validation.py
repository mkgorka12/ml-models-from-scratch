import numpy as np

def validate_predictions_shape(predicted_labels: np.ndarray, 
                       true_labels: np.ndarray) -> None:
    if predicted_labels.shape[0] != true_labels.shape[0]:
        raise ValueError("Labels size and predicted labels size must be the same")

def validate_data(features: np.ndarray, labels: np.ndarray) -> None:
    if features.size == 0:
        raise ValueError("Features are empty")
    
    if features.ndim != 2:
        raise ValueError("Features are expected to be a matrix")
    
    if labels.size == 0:
        raise ValueError("Labels are empty")
    
    if labels.ndim > 1:
        raise ValueError("Labels are expected to be a vector (n, )")

    if features.shape[0] != labels.shape[0]:
        raise ValueError("Feature row sizes and label row sizes must be the same")

def validate_hyperparameters(learning_rate: float, batch_size: int, 
                             number_epochs: int, eps: float) -> None:
    if learning_rate <= 0:
        raise ValueError("Learning rate must be bigger than 0")
    
    if batch_size is not None and batch_size <= 0:
        raise ValueError("Batch size must be bigger than 0")
    
    if number_epochs is not None and number_epochs <= 0:
        raise ValueError("Number of epochs must be bigger than 0")
    
    if eps <= 0:
        raise ValueError("Epsilon (tolerance for loss difference) must be positive")
    
def validate_predict(fitted: bool, features: np.ndarray, weights: np.ndarray) -> None:
    if not fitted:
        raise RuntimeError("Model must be fitted before prediction")
    elif features.shape[1] != weights.shape[0]:
        raise ValueError("Feature dimensions are different from feature dimensions in the training set")
