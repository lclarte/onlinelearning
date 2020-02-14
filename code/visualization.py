# visualization.py
# function to plot various data

import matplotlib.pyplot as plt
import numpy as np

def plot_weights_history(_ps):
    # this way, ps needs not be a numpy array
    ps = np.array(_ps)
    T, K = ps.shape
    absc = np.linspace(1, T, T)
    for k in range(K):
        plt.plot(absc, ps[:, k], label='$p_' + str(k + 1) + '$')
    plt.legend()
    plt.title('History of weights as a function of t')
    plt.xlabel('t')
    plt.ylabel('weights')
    plt.show()