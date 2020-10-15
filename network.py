import numpy as np

class Network():
    def __init__(self, sizes, activation_function, loss_function, seed):
        '''
        Initializes network with random weights and biases, according to passed sizes of layers.
        During inference uses passed activation function. 
        
        If sizes is [20,10,5], then network will consist of input layer is of size 20, 
        one hidden layer sized 10, and output of the network is a vector of length 5.
        '''
        rng = np.random.default_rng(seed)

        self.lossfun = loss_function
        self.afun = activation_function
        self.weights = [rng.standard_normal((x,y)) for x,y in zip(sizes[1:],sizes[:-1])]
        self.biases = [rng.standard_normal(x) for x in sizes[1:]]

    def forward(self, a):
        '''
        Uses 'a' as input of the network, return output
        '''
        for w, b in zip(self.weights, self.biases):
          a = self.afun(np.dot(w, a) + b)
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
                wgradcum += wgrad # divide by len(data) now? (overflows?)
                bgradcum += bgrad

            # descent part
            self.weights = [w - lr * wgrad / len(data) for w, wgrad in zip(self.weights, wgradcum)]
            self.biases = [b - lr * bgrad / len(data) for b, bgrad in zip(self.biases, bgradcum)]

            print(f"Epoch {i} finished. Current accuracy on train data is: {self.evaluate_categorical(data)}")

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
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.afun(z)
            activations.append(activation)
        
        # delta is partial derivative of weighted input z, starting from last layer
        delta = self.lossfun.deriv(activations[-1], y) * self.afun.deriv(zs[-1])
        bgrad[-1] = delta
        wgrad[-1] = activations[-2] * delta.reshape((-1,1))

        for i in range(2, len(activations)):
            delta = np.dot(self.weights[-i+1].transpose(), delta) * self.afun.deriv(zs[-i])
            bgrad[-i] = delta
            wgrad[-i] = activations[-i-1] * delta.reshape((-1,1))
        return wgrad, bgrad

    def evaluate_categorical(self, data):
        results = [(np.argmax(self.forward(x)), y) for (x,y) in data]
        return np.sum([x == y for (x,y) in results])/len(data)
      
    def __call__(self, a):
        return self.forward(a)

    def _empty_grad(self):
        wgrad = np.array([np.zeros(w.shape) for w in self.weights])
        bgrad = np.array([np.zeros(b.shape) for b in self.biases])
        return wgrad, bgrad

