from network import *
from functions import *
from dataprep import *
import pandas as pd
import numpy as np

# for classification select the size of the output layer corresponding to the number of labels?
# for regression select the size of the output layer as 1?

# random number generator seed (null for random seed???)
seed = 123

# set 'reg' for regression and 'cls' for classification
#task_type='cls'
task_type = 'reg'

# number of perceptrons in each layer
# classification: input - space dim, output - number of classes
# regression: input - space dim, output - 1
#net_arch = [2, 10, 3]
net_arch = [1, 3, 3, 1]

# path to train data
#train_data_path = '/home/monika/Pulpit/SN/SN_projekt1/classification/data.three_gauss.test.100.csv'
train_data_path = '/home/monika/Pulpit/SN/SN_projekt1/regression/data.activation.test.100.csv'

#path to test data (to be used in the future)
#currently test data is split
#test_data_path = ''

activation_fun = sigmoid()
loss_fun = SE()

# do you want to see visualisation of the learning process and results
#visualization = True

net = Network(task_type, net_arch, activation_fun, loss_fun, seed)
data = pd.read_csv(train_data_path, sep=',', header=0)

if task_type == 'cls':
    train, test, label_encoder = train_test_from_df_categorical(data, 'cls', 0.9, seed)
elif task_type == 'reg':
    train, test = train_test_from_df_regression(data, 'y', 0.9, seed)

net.GD(train, lr = 1, epochs = 60)

print(f"{'Accuracy' if task_type == 'cls' else 'RMSE'} score on test data: {net.evaluate(test)}")