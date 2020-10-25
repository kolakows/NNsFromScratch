import numpy as np
import math

class loss_function():
    '''
    output_activation is outcome calculated by network, y is true label/value
    '''
    def deriv(self, output_activation, y):
        pass

class activation_function():
    def activ(self, z):
        pass
    def deriv(self, z):
        pass
    def __call__(self, z):
        pass

class sigmoid(activation_function):
    def activ(self, z):
        return 1/(1 + np.exp(-z))
    def deriv(self, z):
        act = self.activ(z)
        return act * (1 - act)
    def __call__(self, z):
        return self.activ(z)

class relu(activation_function):
    def activ(self, z):
        return [max(0, zi) for zi in z]
    def deriv(self, z):
        return [(1 if zi >= 0 else 0) for zi in z]
    def __call__(self, z):
        return self.activ(z)

class param_relu(activation_function):
    def __init__(self, alpha):
        self.alpha = alpha
    def activ(self, z):
        return [(zi if zi >= 0 else self.alpha*zi) for zi in z]
    def deriv(self, z):
        return [(1 if zi >= 0 else self.alpha) for zi in z]
    def __call__(self, z):
        return self.activ(z)

class softmax(activation_function):
    def activ(self, z):
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum()
    def deriv(self, z):
        act = self.activ(z)
        return act * (1 - act)
    def __call__(self, z):
        return self.activ(z)

class linear(activation_function):
    def activ(self, z):
        return z
    def deriv(self, z):
        return np.ones(len(z))
    def __call__(self, z):
        return self.activ(z)

class MSE(loss_function):
    def loss(self, output_activation, y):
        return np.sum((output_activation - y)*(output_activation - y)) / len(y)
    def deriv(self, output_activation, y):
        return (output_activation - y) / len(y)

class SE(loss_function):
    def loss(self, output_activation, y):
        return np.sum((output_activation - y)*(output_activation - y))
    def deriv(self, output_activation, y):
        return output_activation - y

class cross_entropy(loss_function):
    def deriv(self, output_activation, y):
        return - np.divide(y, output_activation) + np.divide(1 - y, 1 - output_activation)

class MAE(loss_function):
    def loss(self, output_activation, y):
        return np.abs(output_activation - y)
    def deriv(self, output_activation, y):
        return [(-1 if z < 0 else 1) for z in (output_activation - y)]

function_dict = {
    # activation functions
    'sigmoid' : sigmoid(),
    'ReLU' : relu(),
    'LeakyReLU' : param_relu(0.01), # LeakyReLU
    'softmax' : softmax(),
    'linear' : linear(),
    # loss functions
    'MSE' : MSE(),
    'SE' : SE(),
    'cross_entropy' : cross_entropy(),
    'MAE' : MAE()
}
