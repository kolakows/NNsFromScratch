import numpy as np

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
        return self.activ(z) * (1 - self.activ(z))
    def __call__(self, z):
        return self.activ(z)

class MSE(loss_function):
    def loss(self, output_activation, y):
        return np.sum((output_activation - y) * (output_activation - y))/len(y)
    def deriv(self, output_activation, y):
        return output_activation - y
