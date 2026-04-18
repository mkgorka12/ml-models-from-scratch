class Activation:
    def __init__(self, activation_funcs):
        self.activation_func = activation_funcs[0]
        self.activation_func_deriv = activation_funcs[1]

        self.s = None

    def forward(self, s):
        self.s = s
        return self.activation_func(s)

    def backward(self, grad):
        return grad * self.activation_func_deriv(self.s)

    def adjust(self, alfa):
        pass
