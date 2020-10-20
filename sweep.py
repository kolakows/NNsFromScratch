from network import *
from functions import *
from dataprep import *
import pandas as pd
import numpy as np
import wandb
import ast
import os
from pathlib import Path



def sweep():

    # load data
    # data = pd.read_csv(data_path, sep=',', header=0)
    # train, test, label_encoder = train_test_from_df_categorical(data, 'cls', 0.9, seed)

    # prepare sweep config
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'data': {
                'values': ['classification/data.three_gauss.test.100.csv']
            },
            'layers': {
                'values': ['[2,3]','[2,2,3]']
            },            
            'loss_function':{
                'values': ['MSE', 'SE']
            },
            'activation_function':{
                'values': ['sigmoid', 'relu']
            },
            'seed':{
                'values': [123]
            },
            'lr':{
                'values': [1,2]
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='network_from_scratch')
    wandb.agent(sweep_id, function=train)

def train():

    local_path = 'C:/MiniProjects/sem2/NeuralNets/'

    config_defaults = dict(
        data = 'classification/data.three_gauss.test.100.csv',
        layers = '[2,3]',
        lr = 1,
        epochs = 30,
        seed = 123,
        loss_function = 'MSE',
        activation_function = 'sigmoid',
    )

    wandb.init(config=config_defaults)
    config = wandb.config

    # prepare path, should work on linux and windows
    data_path = os.path.normpath((os.path.join(local_path,config.data)))

    task_type, _ = os.path.split(config.data)
    if task_type == 'classification':
        task_type = 'cls'
    else:
        task_type = 'reg'

    # read data
    data = pd.read_csv(data_path, sep=',', header=0)
    train, test, label_encoder = train_test_from_df_categorical(data, task_type, 0.9, config.seed)

    net = Network.from_config(config)
    net.GD(train, config.lr, config.epochs, test)
    #print(f'Category accuracy score: {net.evaluate_categorical(test)}')


