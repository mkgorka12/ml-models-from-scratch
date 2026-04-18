import numpy as np


class MaxPooling:
    def __init__(self, filter_shape, strides=(1, 1)):
        self.filter_shape = filter_shape
        self.strides = strides

    def forward(self, input):
        N, H, W = input.shape
        F_h, F_w = self.filter_shape
        S_h, S_w = self.strides

        H_out = (H - F_h) // S_h + 1
        W_out = (W - F_w) // S_w + 1

        output = np.empty((N, H_out, W_out))
        self.idx = np.empty((N, H_out, W_out))

        for row in range(H_out):
            for col in range(W_out):
                patch = input[
                    :, row * S_h : row * S_h + F_h, col * S_w : col * S_w + F_w
                ]

                for i in range(N):
                    self.idx[i, row, col] = np.argmax(patch[i])

                output[:, row, col] = np.max(patch, axis=(1, 2))

        return output

    def backward(self, grad):
        N, H, W = grad.shape
        F_h, F_w = self.filter_shape
        S_h, S_w = self.strides

        H_out = (H - 1) * S_h + F_h
        W_out = (W - 1) * S_w + F_w

        output = np.zeros((N, H_out, W_out))

        for row in range(H):
            for col in range(W):
                highest_in_patch = [
                    np.unravel_index(int(self.idx[i, row, col]), (F_h, F_w))
                    for i in range(N)
                ]

                for i in range(N):
                    output[
                        i,
                        row * S_h + highest_in_patch[i][0],
                        col * S_w + highest_in_patch[i][1],
                    ] = grad[i, row, col]

        return output

    def adjust(self, alpha):
        pass
