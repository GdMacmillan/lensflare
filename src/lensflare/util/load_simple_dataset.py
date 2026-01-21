"""Dataset loading utilities."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons


def load_moons_dataset(n_samples=300, noise=0.2, seed=None, plot=False):
    """Load the moons dataset for binary classification.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate. Default is 300.
    noise : float, optional
        Standard deviation of Gaussian noise. Default is 0.2.
    seed : int, optional
        Random seed for reproducibility.
    plot : bool, optional
        If True, display a scatter plot of the data. Default is False.

    Returns
    -------
    X_train : np.ndarray
        Training data of shape (2, n_samples).
    y_train : np.ndarray
        Labels of shape (1, n_samples).
    """
    if seed is not None:
        np.random.seed(seed)

    X_train, y_train = make_moons(n_samples=n_samples, noise=noise)

    # Optionally visualize the data
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=40, cmap=plt.cm.Spectral)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Moons Dataset')
        plt.show()

    # Transpose to (n_features, n_samples) format
    X_train = X_train.T
    y_train = y_train.reshape((1, y_train.shape[0]))

    return X_train, y_train
