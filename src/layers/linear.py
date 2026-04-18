import numpy as np


class Linear:
    def __init__(self, size, weights_range=(-0.1, 0.1)):
        self.w = np.random.default_rng().uniform(
            weights_range[0], weights_range[1], (size[0], size[1])
        )
        self.x = None
        self.grad = None

    def forward(self, x):
        self.x = x
        return self.w @ x

    def backward(self, grad):
        self.grad = grad
        return self.w.T @ grad

    def adjust(self, alfa):
        self.w -= alfa * self.grad @ self.x.T
