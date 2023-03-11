from matplotlib import pyplot as plt
from utils import load_pickle
import numpy as np

def convolve(x, N=100):
    return np.convolve(x, np.ones(N)/N, mode='valid')
 
def plot_curves():

    data = load_pickle('results/snapshot/nem_ret/1000_logs.pk')
    plt.plot(convolve(data['mean']), label='population mean')
    plt.plot(convolve(data['best']), label='population best')
    plt.legend()
    plt.title("Genetic optimization of NEM")
    plt.xlabel("generation")
    plt.ylabel("avg accuracy over inner loop episode")
    plt.show()


plot_curves()
