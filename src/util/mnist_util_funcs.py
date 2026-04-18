import numpy as np


def mnist_score(prediction, goal):
    pred_labels = np.argmax(prediction, axis=0)
    goal_labels = np.argmax(goal, axis=0)

    correct = np.sum(pred_labels == goal_labels)
    all = goal.shape[1]

    return correct / all


def mnist_one_hot_encoder(y):
    N = y.shape[1]
    res = np.zeros((10, N))
    res[y[0].astype(int), np.arange(N)] = 1.0
    return res
