"""TensorFlow utility functions for neural network operations.

This module provides utility functions for visualization and prediction
that work with TensorFlow 2 models.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_decision_boundary(model, X, y):
    """Plot decision boundary for a binary classifier.

    Parameters
    ----------
    model : callable
        Model that takes input of shape (n_features, m_examples) and returns
        predictions. Can be a TfNNClassifier, a Keras model, or any callable.
    X : np.ndarray
        Data of shape (n_features, m_examples). For 2D visualization,
        n_features should be 2.
    y : np.ndarray
        Labels of shape (1, m_examples).
    """
    plt.figure(figsize=(10, 8))
    plt.title("Model Decision Boundary")

    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])

    # Set min and max values with padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    # Generate a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    grid_input = np.c_[xx.ravel(), yy.ravel()].T

    # Handle different model types
    if hasattr(model, 'predict_proba'):
        Z = model.predict_proba(grid_input)
    elif hasattr(model, 'predict'):
        Z = model.predict(grid_input)
    elif callable(model):
        Z = model(grid_input)
    else:
        raise ValueError("Model must have predict/predict_proba method or be callable")

    # Ensure Z is numpy array and reshape
    if hasattr(Z, 'numpy'):
        Z = Z.numpy()
    Z = np.asarray(Z)
    if Z.ndim > 1:
        Z = Z.flatten()
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.colorbar(label='Probability')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.flatten(), cmap=plt.cm.Spectral,
                edgecolors='black', linewidth=0.5)
    plt.show()


def predict_with_model(model, X, y=None, train=True):
    """Make predictions with a trained model.

    Parameters
    ----------
    model : object
        Trained model with predict or predict_proba method.
    X : np.ndarray
        Input data of shape (n_features, m_examples).
    y : np.ndarray, optional
        True labels of shape (1, m_examples) for accuracy computation.
    train : bool, optional
        If True, print "Training Accuracy", else "Test Accuracy".

    Returns
    -------
    np.ndarray
        Binary predictions of shape (1, m_examples).
    """
    # Get predictions
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
    elif hasattr(model, 'predict'):
        proba = model.predict(X)
    else:
        raise ValueError("Model must have predict or predict_proba method")

    # Convert probabilities to binary predictions
    preds = (proba > 0.5).astype(int)

    # Print accuracy if labels provided
    if y is not None:
        accuracy = np.mean(preds == y)
        label = "Training" if train else "Test"
        print(f"{label} Accuracy: {accuracy:.6f}")

    return preds


def compute_binary_accuracy(y_true, y_pred):
    """Compute binary classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    float
        Accuracy score between 0 and 1.
    """
    return np.mean(y_true == y_pred)
