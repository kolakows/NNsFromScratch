from network import *
from functions import *
from dataprep import *
import pandas as pd
import numpy as np

seed = 123
net = Network([2,3], sigmoid(), MSE(), seed)

data = pd.read_csv(r'/home/igor/NNsScratch/NNsFromScratch/classification/data.three_gauss.test.100.csv', sep=',', header=0)
train, test, label_encoder = train_test_from_df_categorical(data, 'cls', 0.9, seed)
net.GD(train, lr = 1, epochs = 30)
print(f'Category accuracy score: {net.evaluate_categorical(test)}')



