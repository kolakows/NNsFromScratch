import matplotlib.pyplot as plt

def plot_reg_results(test_data, results):
    x_val = [x for (x,y) in test_data]
    y_val = [y for (x,y) in test_data]
    y_nn = [y_n for (y_n,y_real) in results]

    plt.plot(x_val, y_val, '-o', label='actual')
    plt.plot(x_val, y_nn, 'o', label='NN output')

    plt.legend()
    plt.show()