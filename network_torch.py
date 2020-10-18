import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



class Network_torch(nn.Module):
    '''
    Torch version of our simple network, also run on cpu
    '''
    def __init__(self, sizes, activation_function, loss_function, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.layers = nn.ModuleList()
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            # weight initialization using kaiming_uniform for weights (He initialization, derived for relu) 
            # bias initialization using b = inverse of squared input size, uniform(-b,b), sigma \approx 0,6
            self.layers.append(nn.Linear(in_size, out_size).double())
        self.lossfun = loss_function
        self.afun = activation_function

    def forward(self, x):
        for layer in self.layers:
            x = self.afun(layer(x))
        return x
    
    def GD(self, data, lr, epochs):
        optimizer = optim.SGD(self.parameters(), lr)
        for i in range(epochs):
            optimizer.zero_grad()
            cum_loss = None
            for x,y in data:
                # convert x, y to tensors
                x = torch.from_numpy(x)
                y = torch.from_numpy(y)
                # set requires_grad
                x.requires_grad = True
                outputs = self(x)
                loss = self.lossfun(outputs,y)
                # loss.backward() the same as summing losses and calling backward once
                if not cum_loss:
                    cum_loss = loss
                else:
                    cum_loss += loss
            mean_loss = cum_loss/len(data)
            mean_loss.backward()
            optimizer.step()
            print(f"Epoch {i} finished. Current accuracy on train data is: {self.evaluate_categorical(data)}")
            print(f"Cumulated loss for episode is {cum_loss : .3f}")
            #print(f"Mean loss for episode is {mean_loss : .3f}")

    def evaluate_categorical(self, data):
        with torch.no_grad():
            results = [(np.argmax(self.forward(torch.from_numpy(x))), np.argmax(y)) for (x,y) in data]
        return np.sum([x == y for (x,y) in results])/len(data)