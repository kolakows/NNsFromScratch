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
        print(z)
        return [(1 if zi >= 0 else self.alpha) for zi in z]
    def __call__(self, z):
        return self.activ(z)

class softmax(activation_function):
    def activ(self, z):
        e_z = np.exp(z)
        return e_z / e_z.sum()
    def deriv(self, z):
        raise NotImplementedError
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
    def deriv(self, output_activation, y):
        return (output_activation - y) / len(y)

class SE(loss_function):
    def deriv(self, output_activation, y):
        return output_activation - y

class multiclass_cross_entropy(loss_function):
    def deriv(self, output_activation, y):
        pass

class sparse_multiclass_cross_entropy(loss_function):
    def deriv(self, output_activation, y):
        pass