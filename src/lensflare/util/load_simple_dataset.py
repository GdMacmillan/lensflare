import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_moons

def load_moons_dataset(seed=None):
    if seed:
        np.random.seed(seed)
    X_train, y_train = make_moons(n_samples=300, noise=.2)
    # Visualize the data
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=40, cmap=plt.cm.Spectral);
    X_train = X_train.T
    y_train = y_train.reshape((1, y_train.shape[0]))

    return X_train, y_train
