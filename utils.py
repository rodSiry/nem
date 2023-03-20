import pickle
import imageio
import numpy as np

def save_pickle(obj, filename):
    F = open(filename, 'wb')
    pickle.dump(obj, F)
    F.close()

def load_pickle(filename):
    F = open(filename, 'rb')
    obj = pickle.load(F)
    F.close()
    return obj

def make_gif(filepath, frames):
    with imageio.get_writer(filepath, mode='I') as writer:
        for frame in frames:
            writer.append_data(frame)

def accuracy(y, Y):
    Y = Y.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y = np.argmax(y, axis=-1)
    acc = (y == Y).astype(int)
    return acc

def convolve(x, N=1):
    return np.convolve(x, np.ones(N)/N, mode='valid')
 

