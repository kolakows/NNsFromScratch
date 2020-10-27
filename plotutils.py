import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_reg_results(test_data, results):
    x_val = [x for (x,y) in test_data]
    y_val = [y for (x,y) in test_data]
    y_nn = [y_n for (y_n,y_real) in results]

    plt.plot(x_val, y_val, '-o', label='actual')
    plt.plot(x_val, y_nn, 'o', label='NN output')

    plt.legend()
    plt.show()

def plot_loss_function(loss, plot_title):
    ax = sns.lineplot(y = loss, x = np.arange(len(loss)))
    ax.set_title(plot_title)
    plt.show()
    plt.close()

def plot_classification_results(network, data):
    _prepare_classification_plot(network, data)
    plt.show()

def _prepare_classification_plot(network, data):
    x = np.arange(0,1,0.01)
    y = x
    xx, yy = np.meshgrid(x,y)

    net_outputs_grid = [network.forward([x,y]) for x,y in zip(xx.ravel(),yy.ravel())]
    df = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel(), 'cls': np.argmax(net_outputs_grid, axis= 1)})
    sns.scatterplot(data = df, x = 'x', y = 'y', hue = 'cls', size = 0.01, palette='deep', edgecolor=None)

    true_labels = [y for x,y in data]
    x_coords = [x[0] for x,y in data]
    y_coords = [x[1] for x,y in data]
    df = pd.DataFrame({'cls': np.argmax(true_labels, axis = 1), 'x': x_coords, 'y': y_coords})
    ax = sns.scatterplot(data = df, x = 'x', y = 'y', hue = 'cls', size = 0.1, palette='deep')
    ax.legend([],[], frameon=False)

def get_data_plot(data, network):
    if network.task == 'cls':
        _prepare_classification_plot(network, data)
        # x_val = [x for (x,y) in data]
        # y_val = [np.argmax(y) for (x,y) in data]
        # y_nn = [y_n for (y_n,y_real) in results]

        # df = pd.DataFrame({'coords': x_val, 'Actual': y_val, 'NN output': y_nn})
        # df = df.melt(id_vars=['coords'], var_name='data source', value_name = 'class')
        # df[['x','y']] = pd.DataFrame(df['coords'].tolist(), index = df.index)
        # df['class'] = df['class'].astype(str)

        # c_num = len(df['class'].unique())

        # ax = sns.scatterplot(data = df, x = 'x', y = 'y', hue = 'class', size = 'data source',
        #                     palette = sns.color_palette()[:c_num], sizes = (50,150))
        

        # ax = sns.scatterplot(data = df[df['data source'] == 'Actual'], x = 'x', y = 'y', hue = 'class', 
        # palette = sns.color_palette()[c_num: 2*c_num]) #, style='data source')
        return plt
    else:
        score, results = network.evaluate(data)
        x_val = [x for (x,y) in data]
        y_val = [y for (x,y) in data]
        y_nn = [y_n for (y_n,y_real) in results]

        df = pd.DataFrame({'x': x_val, 'Actual': y_val, 'NN output': y_nn})
        df = df.melt(id_vars=['x'], var_name='data source', value_name = 'y')
        df = df.explode('x').explode('y')

        ax = sns.scatterplot(data = df, x = 'x', y = 'y', hue = 'data source', edgecolor=None)
        return plt
