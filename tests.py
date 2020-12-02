from network import *
from functions import *
from dataprep import *
import pandas as pd
from plotutils import *
import numpy as np

# for classification select the size of the output layer corresponding to the number of labels
# for regression select the size of the output layer as 1

# random number generator seed (None for random seed)
seed = 4

# set 'reg' for regression and 'cls' for classification
task_type='cls'
#task_type = 'reg'

# number of perceptrons in each layer
# classification: input - space dim, output - number of classes
# regression: input - space dim, output - 1
net_arch = [2, 8, 8, 4, 4]

# path to train data
train_data_path = r'C:\MiniProjects\sem2\NeuralNets\SN_projekt1_test\Classification\data.circles.train.1000.csv'
test_data_path = r'C:\MiniProjects\sem2\NeuralNets\SN_projekt1_test\Classification\data.circles.test.1000.csv'

activation_fun = relu()
loss_fun = MSE()

net = Network(task_type, net_arch, activation_fun, loss_fun, seed)

train_data = pd.read_csv(train_data_path, sep=',', header=0)
test_data = pd.read_csv(test_data_path, sep=',', header=0)

if task_type == 'cls':
    train_data, test_data, _, _ = train_test_from_files_categorical(train_data, test_data, 'cls', seed)
elif task_type == 'reg':
    train_data, test_data, _ = train_test_from_files_regression(train_data, test_data, 'y', seed)

# do you want to see visualisation of the learning process and results
visualization = True
net.SGD(train_data, lr = 1, epochs = 200, log_accuracy=False, plot_loss=True)

test_data.sort(key = lambda val: val[0][0])
score, results = net.evaluate(test_data)

print(f"{'Accuracy' if task_type == 'cls' else 'RMSE'} score on test data: {score}")

if visualization and task_type == 'reg':
    plot_reg_results(test_data, results)
if task_type == 'cls':
    plot_classification_results(net, test_data)