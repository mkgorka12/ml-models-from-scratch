import numpy as np


class Dropout:
    def __init__(self, enabled_neurons_ratio=0.5):
        self.enabled_neurons_ratio = enabled_neurons_ratio
        self.mask = None

    def forward(self, x):
        mask = np.random.binomial(n=1, p=self.enabled_neurons_ratio, size=x.shape)
        self.mask = mask
        return x * mask / self.enabled_neurons_ratio

    def backward(self, grad):
        return grad * self.mask / self.enabled_neurons_ratio

    def adjust(self, alfa):
        pass
