from network import *
from functions import *
import pandas as pd
import numpy as np

net = Network([2, 2], sigmoid(), MSE(), 123)

#, dtype={'x':np.float64, 'y':np.float64, 'cls':np.float64}

data = pd.read_csv(r'C:\MiniProjects\sem2\NeuralNets\classification\data.simple.test.100.csv', sep=',', header=0)
data = [(x,y) for x, y in zip(data.loc[:, data.columns != 'cls'].to_numpy(), data.loc[:,'cls'].to_numpy())]
net.GD(data[:90], lr = 1, epochs = 30)
print(f'Category accuracy score: {net.evaluate_categorical(data[90:])}')


