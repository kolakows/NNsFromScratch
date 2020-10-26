from network import *
from functions import *
from dataprep import *
import pandas as pd
import numpy as np
import wandb
import ast
import os
from pathlib import Path



def sweep(sweep_config):

    # load data
    # data = pd.read_csv(data_path, sep=',', header=0)
    # train, test, label_encoder = train_test_from_df_categorical(data, 'cls', 0.9, seed)

    sweep_id = wandb.sweep(sweep_config, project='network_from_scratch')
    wandb.agent(sweep_id, function=train)

def train():

    local_path = 'C:/MiniProjects/sem2/NeuralNets/'

    config_defaults = dict(
        data = 'classification/data.three_gauss.test.100.csv',
        layers = '[2,4,3]',
        lr = 1,
        epochs = 30,
        seed = 123,
        loss_function = 'cross_entropy',
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
 
    if task_type == 'cls':
        train, test, label_encoder, feature_scalers = train_test_from_df_categorical(data, 'cls', 0.9, config.seed)
    elif task_type == 'reg':
        train, test, feature_scalers = train_test_from_df_regression(data, 'y', 0.9, config.seed)


    net = Network.from_config(config)
    net.GD(train, config.lr, config.epochs, test, True, False)

    # log final accuracy
    log_data = {}
    score, results = net.evaluate(train)  
    log_data[f"{'Accuracy' if net.task == 'cls' else 'RMSE'} on train data"] = score
    score, results = net.evaluate(test)
    log_data[f"{'Accuracy' if net.task == 'cls' else 'RMSE'} on test data"] = score
    wandb.log(log_data, step=config.epochs)


    wandb.log({'Train data plot': wandb.Image(get_data_plot(train, net))}, step=config.epochs)
    plt.close()
    wandb.log({'Test data plot': wandb.Image(get_data_plot(test, net))}, step=config.epochs)
    plt.close()
    #print(f'Category accuracy score: {net.evaluate_categorical(test)}')