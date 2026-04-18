import numpy as np


def load_mnist_labels(file_path):
    content = None
    with open(file_path, "rb") as f:
        content = f.read()

    labels = np.frombuffer(content, dtype=np.uint8, offset=8)
    labels = labels.reshape(1, labels.shape[0])

    return labels


def load_mnist_images(file_path):
    content = None
    with open(file_path, "rb") as f:
        content = f.read()

    features_number = int.from_bytes(content[4:8], "big")

    features = np.frombuffer(content, dtype=np.uint8, offset=16).astype(np.float32)
    features = features.reshape(features_number, 28, 28) / 255.0

    return features
