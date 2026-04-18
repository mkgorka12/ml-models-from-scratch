import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(x.dtype)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


def tanh(x):
    x = np.clip(x, -20, 20)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_deriv(x):
    return 1 - np.power(x, 2)


def softmax(x):
    return np.exp(x) / sum(np.exp(x))
