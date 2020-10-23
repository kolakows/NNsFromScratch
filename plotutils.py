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

def get_data_plot(data, network):
    if network.task == 'cls':
        score, results = network.evaluate(data)

        x_val = [x for (x,y) in data]
        y_val = [np.argmax(y) for (x,y) in data]
        y_nn = [y_n for (y_n,y_real) in results]

        df = pd.DataFrame({'coords': x_val, 'Actual': y_val, 'NN output': y_nn})
        df = df.melt(id_vars=['coords'], var_name='data source', value_name = 'class')
        df[['x','y']] = pd.DataFrame(df['coords'].tolist(), index = df.index)
        df['class'] = df['class'].astype(str)

        c_num = len(df['class'].unique())

        ax = sns.scatterplot(data = df, x = 'x', y = 'y', hue = 'class', size = 'data source',
                            palette = sns.color_palette()[:c_num], sizes = (50,150))
        
        
        # ax = sns.scatterplot(data = df[df['data source'] == 'Actual'], x = 'x', y = 'y', hue = 'class', 
        # palette = sns.color_palette()[c_num: 2*c_num]) #, style='data source')
        return plt
    else:
        x_val = [x for (x,y) in data]
        y_val = [y for (x,y) in data]
        y_nn = [y_n for (y_n,y_real) in results]

        plt.plot(x_val, y_val, '-o', label='Actual')
        plt.plot(x_val, y_nn, 'o', label='NN output')
        return plt
