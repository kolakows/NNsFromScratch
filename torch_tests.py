from dataprep import *
from torch_network import *
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

seed = 123
net = Network_torch([2,3], torch.sigmoid, nn.MSELoss(reduction='sum'), seed)

data = pd.read_csv(r'/home/igor/NNsScratch/NNsFromScratch/classification/data.three_gauss.test.100.csv', sep=',', header=0)
train, test, label_encoder = train_test_from_df_categorical(data, 'cls', 0.9, seed)
net.GD(train, lr = 1, epochs = 100)
print(f'Category accuracy score: {net.evaluate_categorical(test)}')