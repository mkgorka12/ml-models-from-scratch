import numpy as np
from src.layers.activation import Activation
from src.layers.dropout import Dropout
from src.util.activation_funcs import softmax


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def _shuffle_dataset(self, x, y):
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        return x[indices], y[:, indices]

    def _forward_pass(self, x, training=True):
        out = x
        for layer in self.layers:
            if not training and type(layer) == Dropout:
                continue
            out = layer.forward(out)
        return out

    def _backward_pass(self, grad):
        for idx, layer in enumerate(reversed(self.layers)):
            if (
                idx == 0
                and type(self.layers[-1]) == Activation
                and self.layers[-1].activation_func == softmax
            ):
                continue
            elif (
                idx == 1
                and type(self.layers[-1]) == Activation
                and self.layers[-1].activation_func == softmax
            ):
                grad = layer.backward(grad) / grad.shape[1]
                continue
            grad = layer.backward(grad)
        return grad

    def _update_weights(self, alfa):
        for layer in self.layers:
            layer.adjust(alfa)

    def fit(self, x, y, alpha=0.01, number_epochs=50, batch_size=None, score=None):
        batch_size = x.shape[0] if batch_size is None else batch_size
        print_score = score is not None
        samples_number = x.shape[2]
        number_batches = int(np.ceil(samples_number / batch_size))

        for epoch_idx in range(number_epochs):
            x, y = self._shuffle_dataset(x, y)

            for batch_idx in range(number_batches):
                batch_x = x[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_y = y[:, batch_idx * batch_size : (batch_idx + 1) * batch_size]

                pred = self._forward_pass(batch_x)

                grad = 2 / batch_size * (pred - batch_y)
                grad = self._backward_pass(grad)

                self._update_weights(alpha)

            if print_score:
                print(f"Epoch {epoch_idx}, accuracy: {score(self.predict(x), y)}")

    def predict(self, x):
        return self._forward_pass(x, False)
