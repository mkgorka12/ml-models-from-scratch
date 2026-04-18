import numpy as np


def convolution(img, filter, stride=1, pad_width=0):
    res_width = int((img.shape[0] - filter.shape[0] + 2 * pad_width) / stride + 1)
    res_height = int((img.shape[1] - filter.shape[1] + 2 * pad_width) / stride + 1)

    img = np.pad(img, pad_width)
    res = np.zeros((res_height, res_width))

    for row in range(0, img.shape[0] - filter.shape[0] + 1, stride):
        for col in range(0, img.shape[1] - filter.shape[1] + 1, stride):
            res[row, col] = np.sum(
                img[row : row + filter.shape[0], col : col + filter.shape[1]] * filter
            )
    return res
