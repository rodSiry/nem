from matplotlib import pyplot as plt
from utils import load_pickle, convolve
import numpy as np

def plot_curves(path):

    data = load_pickle(path)
    plt.plot(convolve(data['mean']), label='population mean')
    plt.plot(convolve(data['best']), label='population best')
    plt.legend()
    plt.title("Genetic optimization of NEM")
    plt.xlabel("generation")
    plt.ylabel("avg accuracy over inner loop episode")


plot_curves('results/snapshot/nem_states/1000_logs.pk')
plot_curves('results/snapshot/nem_small/10000_logs.pk')
plot_curves('results/snapshot/nem_small/2000_logs.pk')
plt.show()
