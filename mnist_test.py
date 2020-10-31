from network import *
from functions import *
from dataprep import *
import pandas as pd
from plotutils import *
import numpy as np
from mnist import MNIST


mndata = MNIST('/home/monika/Pulpit/SN/mnist')
img, labels = mndata.load_training()
img_test, labels_test = mndata.load_testing()

seed = 1
task_type='cls'
net_arch = [784, 100, 10]
activation_fun = relu()
loss_fun = cross_entropy()

net = Network(task_type, net_arch, activation_fun, loss_fun, seed)

train = parse_mnist(img, labels)
test = parse_mnist(img_test, labels_test)

net.GD(train, lr = 0.5, epochs = 15, log_accuracy=True, plot_loss=True)

score, results = net.evaluate(test)

print(f"Accuracy score on test data: {score}")