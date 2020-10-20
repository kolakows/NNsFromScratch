import numpy as np
from functions import *
import math

class Network():
    def __init__(self, task_type, network_size, activation_function, loss_function, seed, set_biases = True):
        '''
        Initializes network with random weights and biases, according to passed sizes of layers.
        During inference uses passed activation function. 
        
        If sizes is [20,10,5], then network will consist of input layer is of size 20, 
        one hidden layer sized 10, and output of the network is a vector of length 5.
        '''
        rng = np.random.default_rng(seed)
        #network_size.append(network_size[-1])

        self.rng = rng
        self.lossfun = loss_function
        self.afun = activation_function
        self.afun_output = softmax() if task_type == 'cls' else linear()
        self.layer_count = len(network_size)
        self.weights = [rng.standard_normal((x,y))/np.sqrt(y) for x,y in zip(network_size[1:],network_size[:-1])]
        self.set_biases = set_biases
        if set_biases:
            self.biases = [rng.standard_normal(x) for x in network_size[1:]]
        else:
            self.biases = [np.zeros(x) for x in network_size[1:]]
        self.task = task_type

    def forward(self, a):
        '''
        Uses 'a' as input of the network, return output
        '''
        for w, b, layer_no in zip(self.weights, self.biases, range(1, self.layer_count)):
            if layer_no < self.layer_count - 1 :
                a = self.afun(np.dot(w, a) + b)
            else:
                a = self.afun_output(np.dot(w, a) + b) 
        return a

    # GD will take long time to compute for large datasets, compared to SGD
    def GD(self, data, lr, epochs):
        '''
        Implements full gradient descent over all data, repeats for x epochs
        '''
        for i in range(epochs):
            # calculate gradient part
            wgradcum, bgradcum = self._empty_grad()
            for x,y in data:
                wgrad, bgrad = self.backprop(x,y)
                wgradcum = [old_w + w for old_w, w in zip(wgradcum, wgrad)] # divide by len(data) now? (overflows?)
                bgradcum = [old_b + b for old_b, b in zip(bgradcum, bgrad)]

            # descent part
            self.weights = [w - lr * wgrad / len(data) for w, wgrad in zip(self.weights, wgradcum)]
            if self.set_biases:
                self.biases = [b - lr * bgrad / len(data) for b, bgrad in zip(self.biases, bgradcum)]

            print(f"Epoch {i} finished. Current {'accuracy' if self.task == 'cls' else 'RMSE'} on train data is: {self.evaluate(data)}")

    def backprop(self, x, y):
        '''
        returns (wgrad, bgrad) which are layer by layer gradients of weights and biases wrt. cost function,
        array structure of (wgrad, bgrad) matches that of self.weights, self.biases
        '''
        wgrad, bgrad = self._empty_grad()
        
        activation = x
        activations = [x] # list to store activations, layer by layer
        zs = [] # list to store weighted inputs, layer by layer

            
        # feed forward
        for w, b, layer_no in zip(self.weights, self.biases, range(1, self.layer_count)):
            z = np.dot(w, activation) + b
            zs.append(z)
            if layer_no < self.layer_count - 1 :
                activation = self.afun(z)
            else:
                activation = self.afun_output(z)
            activations.append(activation)
        
        # delta is partial derivative of weighted input z, starting from last layer
        delta = self.lossfun.deriv(activations[-1], y) * self.afun_output.deriv(zs[-1])
        bgrad[-1] = delta
        wgrad[-1] = activations[-2] * delta.reshape((-1,1))

        for i in range(2, len(activations)):
            delta = np.dot(self.weights[-i+1].transpose(), delta) * self.afun.deriv(zs[-i])
            bgrad[-i] = delta
            wgrad[-i] = activations[-i-1] * delta.reshape((-1,1))
        return wgrad, bgrad

    def evaluate(self, data):
        if self.task == 'cls':
            #accuracy
            results = [(np.argmax(self.forward(x)), np.argmax(y)) for (x,y) in data]
            return np.sum([x == y for (x,y) in results])/len(data)
        else:
            #root mean squared error
            results = [(self.forward(x)[0], y) for (x,y) in data]
            return math.sqrt(np.sum((x - y)**2 for (x,y) in results) / len(results))
      
    def __call__(self, a):
        return self.forward(a)

    def _empty_grad(self):
        wgrad = [np.zeros(w.shape) for w in self.weights]
        bgrad = [np.zeros(b.shape) for b in self.biases]
        return wgrad, bgrad

