import numpy as np
from functions import *
import math
import os
import ast
import wandb

class Network():
    def __init__(self, task_type, sizes, activation_function, loss_function, seed, set_biases = True):
        '''
        Initializes network with random weights and biases, according to passed sizes of layers.
        During inference uses passed activation function. 
        
        If sizes is [20,10,5], then network will consist of input layer is of size 20, 
        one hidden layer sized 10, and output of the network is a vector of length 5.
        '''
        rng = np.random.default_rng(seed)
        network_size = sizes
        network_size.append(network_size[-1])

        self.rng = rng
        self.lossfun = loss_function
        self.afun = activation_function
        self.afun_output = activation_function if task_type == 'cls' else linear() #change to softmax for classification in the future
        self.layer_count = len(network_size)
        self.weights = [rng.standard_normal((x,y))/np.sqrt(y) for x,y in zip(network_size[1:],network_size[:-1])]
        self.set_biases = set_biases
        if set_biases:
            self.biases = [rng.standard_normal(x) for x in network_size[1:]]
        else:
            self.biases = [np.zeros(x) for x in network_size[1:]]
        self.task = task_type

    @classmethod
    def from_config(cls, config):
        # extract task type from data path
        task_type, _ = os.path.split(config.data)
        if task_type == 'classification':
            task_type = 'cls'
        else:
            task_type = 'reg'

        activation_function = function_dict[config.activation_function]
        loss_function = function_dict[config.loss_function]
        layers = ast.literal_eval(config.layers)
        return cls(task_type, layers, activation_function, loss_function, config.seed)

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
    def GD(self, train_data, lr, epochs, test_data = None):
        '''
        Implements full gradient descent over all data, repeats for x epochs
        '''
        for i in range(epochs):
            # calculate gradient part
            wgradcum, bgradcum = self._empty_grad()
            for x,y in train_data:
                wgrad, bgrad = self.backprop(x,y)
                wgradcum += wgrad # divide by len(data) now? (overflows?)
                bgradcum += bgrad

            # descent part
            self.weights = [w - lr * wgrad / len(train_data) for w, wgrad in zip(self.weights, wgradcum)]
            if self.set_biases:
                self.biases = [b - lr * bgrad / len(train_data) for b, bgrad in zip(self.biases, bgradcum)]

            print(f"Epoch {i} finished. Current {'accuracy' if self.task == 'cls' else 'RMSE'} on train data is: {self.evaluate(train_data)}")
            # log epoch
            wandb.log({'epoch': i, 'train loss': self.calculate_loss(train_data)})
            if test_data:
                wandb.log({'test loss': self.calculate_loss(test_data)})

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

    def calculate_loss(self, data):
        results = []
        if self.task == 'cls':
            results = [(np.argmax(self.forward(x)), np.argmax(y)) for (x,y) in data]
        else:
            results = [(self.forward(x)[0], y) for (x,y) in data]
        return np.sum([self.lossfun.loss(output, y) for output, y in results])
      
    def __call__(self, a):
        return self.forward(a)

    def _empty_grad(self):
        wgrad = np.array([np.zeros(w.shape) for w in self.weights])
        bgrad = np.array([np.zeros(b.shape) for b in self.biases])
        return wgrad, bgrad

