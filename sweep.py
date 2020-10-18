from network import *
from functions import *
from dataprep import *
import pandas as pd
import numpy as np
import wandb
import ast

def sweep(data_path, seed):

    # load data
    # data = pd.read_csv(data_path, sep=',', header=0)
    # train, test, label_encoder = train_test_from_df_categorical(data, 'cls', 0.9, seed)

    # prepare sweep config
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'layers': {
                'values': ['[2,3]','[2,2,3]']
            },
            'lr':{
                'values': [1,2]
            }
            # 'loss_function':{
            #     'values': ['MSE()', 'MSE2()']
            # },
            # 'activation_function': 'sigmoid()',
            # 'train': 'train',
            # 'test': 'test'
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='network_from_scratch')
    wandb.agent(sweep_id, function=train)

def train():
    data = pd.read_csv(data_path, sep=',', header=0)
    train, test, label_encoder = train_test_from_df_categorical(data, 'cls', 0.9, seed)

    config_defaults = dict(
        layers = f'[{len(train[0][0])}, {len(train[0][1])}]',
        lr = 1
        # loss_function = MSE(),
        # activation_function = sigmoid(),
        # train = None,
        # test = None
    )


    wandb.init(config=config_defaults)
    config = wandb.config
    
    net = Network(ast.literal_eval(config.layers), sigmoid(), MSE(), seed)
    net.GD(train, lr = config.lr, epochs = 30)
    print(f'Category accuracy score: {net.evaluate_categorical(test)}')


if __name__ == "__main__":
    data_path = r'/home/igor/NNsScratch/NNsFromScratch/classification/data.three_gauss.test.100.csv'
    seed = 123
    sweep(data_path, seed)

