import numpy as np


class Flatten:
    def forward(self, x):
        # x.shape -> (N, number_patches, filters)
        self.shape = x.shape

        # output.shape before .T -> (N, number_patches * filters)
        # output.shape.T -> (number_patches * filters, N)
        return x.reshape(x.shape[0], -1).T

    def backward(self, grad):
        # grad.shape -> (number_patches * filters, N)
        # grad.T.shape -> (N, number_patches * filters)
        # self.shape -> (N, number_patches, filters)
        # grad.T.reshape(self.shape) -> (N, number_patches, filters)
        return grad.T.reshape(self.shape)

    def adjust(self, alfa):
        pass
