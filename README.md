# NNsFromScratch

Implementation of neural networks using numpy. Network has attached W&B to it, example training results on 2d dataset can be found here:

https://wandb.ai/krucjator/network_from_scratch?workspace=user-krucjator

# Usage

Script `tests.py` shows example usage of network. There are multiple parameteres of the network that can be adjusted before training. \
Script `mnist_test.py` trains network on MNIST dataset.

Tunable hyperparameters in order of appeareance in tests.py

- Visualization type.
```python
task_type='cls' # 'cls' for classification task, 'reg' for regression
```
- Network architecture, i.e. number of neurons in each layer. First one should be of input size, last of output size.
```python
net_arch = [2, 8, 8, 4, 4]
```
See function_dict in `functions.py` for available activation and loss functions.
- Activation function, it will be set for all layers.
```python
activation_fun = relu()
```
- Loss function.
```python
loss_fun = MSE()
```

- Learning rate, number of epochs, batch size
```python
net = Network(task_type, net_arch, activation_fun, loss_fun, seed)
net.SGD(train_data, lr = 1, epochs = 500, batch_size = 50)
```

# Implementation

Core code of the network (layers' definition, backprop) follows Michael Nielsen's great tutorial which can be found [here](http://neuralnetworksanddeeplearning.com/).

# Github branch ''PyTorch''

On separate branch called PyTorch, there is a short implementation of the same network using PyTorch library.
