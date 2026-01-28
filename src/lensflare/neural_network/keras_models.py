"""TensorFlow 2 / Keras implementation of neural network binary classifier.

This module provides an educational implementation of a neural network using
TensorFlow 2's low-level GradientTape API. This approach explicitly shows the
forward and backward propagation steps, making it ideal for learning.

On Apple Silicon Macs, install tensorflow-metal for GPU acceleration:
    pip install tensorflow-metal
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ..util import random_mini_batches


def _is_apple_silicon():
    """Check if running on Apple Silicon."""
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _is_metal_installed():
    """Check if tensorflow-metal is installed."""
    import importlib.util
    return importlib.util.find_spec("tensorflow_metal") is not None


def _install_tensorflow_metal():
    """Install tensorflow-metal plugin for Apple Silicon GPU support."""
    import subprocess
    import sys

    print("Installing tensorflow-metal for Apple Silicon GPU support...")
    try:
        # Try uv first (faster), fall back to pip
        result = subprocess.run(
            ["uv", "pip", "install", "tensorflow-metal"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            # Fall back to pip
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "tensorflow-metal"],
                check=True,
                capture_output=True
            )
        print("tensorflow-metal installed successfully.")
        print("NOTE: Restart your Python kernel/session to enable Metal GPU.")
        return True
    except Exception as e:
        print(f"Failed to install tensorflow-metal: {e}")
        print("  Manual install: pip install tensorflow-metal")
        return False


def check_gpu_available(auto_install_metal=True):
    """Check if GPU (including Apple Metal) is available for TensorFlow.

    Parameters
    ----------
    auto_install_metal : bool, default=True
        If True and running on Apple Silicon without tensorflow-metal,
        automatically install the Metal plugin.

    Returns
    -------
    bool
        True if GPU is available, False otherwise.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow GPU devices available: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        return True

    # Check if we're on Apple Silicon without tensorflow-metal
    if _is_apple_silicon():
        if not _is_metal_installed():
            print("No GPU available. Running on CPU.")
            print("  Apple Silicon detected - Metal GPU acceleration available.")
            if auto_install_metal:
                _install_tensorflow_metal()
            else:
                print("  Install tensorflow-metal for GPU acceleration:")
                print("    pip install tensorflow-metal")
        else:
            print("No GPU available. Running on CPU.")
            print("  tensorflow-metal is installed but GPU not detected.")
            print("  Restart your Python kernel/session to enable Metal GPU.")
    else:
        print("No GPU available. Running on CPU.")
    return False


def configure_gpu_memory_growth():
    """Configure GPU memory growth to avoid allocating all memory at once."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth configuration failed: {e}")


class DenseLayer(tf.keras.layers.Layer):
    """Custom dense layer with explicit weight initialization.

    This layer demonstrates the low-level construction of a fully connected
    layer with configurable initialization and regularization.

    Parameters
    ----------
    units : int
        Number of neurons in this layer.
    l2_lambda : float, optional
        L2 regularization strength. Default is 0.0 (no regularization).
    activation : str or None, optional
        Activation function: 'relu', 'sigmoid', or None. Default is None.
    seed : int, optional
        Random seed for weight initialization.
    """

    def __init__(self, units, l2_lambda=0.0, activation=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.l2_lambda = l2_lambda
        self.activation_name = activation
        self.seed = seed

    def build(self, input_shape):
        # Xavier/Glorot initialization
        initializer = tf.keras.initializers.GlorotUniform(seed=self.seed)

        # L2 regularization if specified
        regularizer = None
        if self.l2_lambda > 0:
            regularizer = tf.keras.regularizers.l2(self.l2_lambda)

        self.w = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer=initializer,
            regularizer=regularizer,
            trainable=True,
        )
        self.b = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )

        # Set up activation function
        if self.activation_name == 'relu':
            self.activation = tf.nn.relu
        elif self.activation_name == 'sigmoid':
            self.activation = tf.nn.sigmoid
        else:
            self.activation = None

    def call(self, inputs, training=False):
        # Linear transformation: Z = XW + b
        z = tf.matmul(inputs, self.w) + self.b

        # Apply activation if specified
        if self.activation is not None:
            return self.activation(z)
        return z


class BinaryClassifierNN(tf.keras.Model):
    """Neural Network binary classifier using TF2 low-level API.

    This model demonstrates educational neural network implementation using
    explicit GradientTape for backpropagation. The architecture consists of
    fully connected layers with ReLU activation and optional dropout,
    followed by a sigmoid output layer.

    Parameters
    ----------
    layers_dims : list of int
        Dimensions of each layer. First element is input size, last must be 1
        for binary classification.
    l2_lambda : float, optional
        L2 regularization strength. Default is 0.0.
    dropout_rate : float, optional
        Dropout rate (1 - keep_prob). Default is 0.0 (no dropout).
    seed : int, optional
        Random seed for reproducibility.

    Example
    -------
    >>> model = BinaryClassifierNN([2, 64, 32, 1], l2_lambda=0.01, dropout_rate=0.2)
    >>> model.build((None, 2))
    >>> print(model.summary())
    """

    def __init__(self, layers_dims, l2_lambda=0.0, dropout_rate=0.0, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.layers_dims = layers_dims
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.seed = seed

        # Build hidden layers with ReLU activation
        self.hidden_layers = []
        self.dropout_layers = []

        for i, units in enumerate(layers_dims[1:-1]):
            layer = DenseLayer(
                units=units,
                l2_lambda=l2_lambda,
                activation='relu',
                seed=seed + i if seed else None,
                name=f'dense_{i+1}'
            )
            self.hidden_layers.append(layer)

            if dropout_rate > 0:
                self.dropout_layers.append(tf.keras.layers.Dropout(dropout_rate))
            else:
                self.dropout_layers.append(None)

        # Output layer with sigmoid activation
        self.output_layer = DenseLayer(
            units=layers_dims[-1],
            l2_lambda=0.0,  # No regularization on output layer
            activation='sigmoid',
            seed=seed + len(layers_dims) if seed else None,
            name='output'
        )

    def build(self, input_shape):
        """Build the model by building all layers."""
        # Build hidden layers
        shape = input_shape
        for i, layer in enumerate(self.hidden_layers):
            layer.build(shape)
            shape = (shape[0], layer.units)
            if self.dropout_layers[i] is not None:
                self.dropout_layers[i].build(shape)

        # Build output layer
        self.output_layer.build(shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Forward propagation through the network.

        Parameters
        ----------
        inputs : tf.Tensor
            Input data of shape (batch_size, n_features).
        training : bool
            Whether in training mode (affects dropout).

        Returns
        -------
        tf.Tensor
            Output probabilities of shape (batch_size, 1).
        """
        x = inputs

        # Pass through hidden layers
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x, training=training)
            if training and self.dropout_layers[i] is not None:
                x = self.dropout_layers[i](x, training=training)

        # Output layer
        return self.output_layer(x, training=training)

    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss with optional L2 regularization.

        Parameters
        ----------
        y_true : tf.Tensor
            True labels of shape (batch_size, 1).
        y_pred : tf.Tensor
            Predicted probabilities of shape (batch_size, 1).

        Returns
        -------
        tf.Tensor
            Scalar loss value.
        """
        # Binary cross-entropy loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        loss = bce(y_true, y_pred)

        # Add regularization losses if any
        if self.losses:
            loss += tf.add_n(self.losses)

        return loss


class TfNNClassifier:
    """sklearn-style wrapper for BinaryClassifierNN with TF2 GradientTape training.

    This classifier provides a familiar fit/predict API while using TensorFlow 2's
    low-level GradientTape for explicit gradient computation, making it ideal for
    educational purposes.

    Parameters
    ----------
    layers_dims : list of int
        Network architecture. First element is input features, last must be 1.
    optimizer : str, optional
        Optimizer: 'gd' (SGD), 'momentum', or 'adam'. Default is 'gd'.
    alpha : float, optional
        Learning rate. Default is 0.0007.
    mini_batch_size : int, optional
        Mini-batch size. Default is 64.
    lambd : float, optional
        L2 regularization strength. Default is None (no regularization).
    keep_prob : float, optional
        Dropout keep probability (1 - dropout_rate). Default is 1.0 (no dropout).
    beta : float, optional
        Momentum coefficient (for momentum optimizer). Default is 0.9.
    beta1 : float, optional
        Adam first moment decay. Default is 0.9.
    beta2 : float, optional
        Adam second moment decay. Default is 0.999.
    epsilon : float, optional
        Adam epsilon for numerical stability. Default is 1e-8.
    num_epochs : int, optional
        Number of training epochs. Default is 10000.
    print_cost : bool, optional
        Print cost during training. Default is True.

    Attributes
    ----------
    model_ : BinaryClassifierNN
        The underlying Keras model after fitting.
    costs_ : list of float
        Training loss history.
    parameters_ : dict
        Dictionary of trained weight arrays (for compatibility).

    Example
    -------
    >>> from lensflare.classification import TfNNClassifier
    >>> clf = TfNNClassifier([2, 64, 32, 1], optimizer='adam', lambd=0.01)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(self, layers_dims, optimizer='gd', alpha=0.0007, mini_batch_size=64,
                 lambd=None, keep_prob=1.0, beta=0.9, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, num_epochs=10000, print_cost=True):

        self.layers_dims = layers_dims
        self.optimizer_name = optimizer
        self.alpha = alpha
        self.mini_batch_size = mini_batch_size
        self.lambd = lambd if lambd is not None else 0.0
        self.keep_prob = keep_prob
        self.dropout_rate = 1.0 - keep_prob if keep_prob < 1.0 else 0.0
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.num_epochs = num_epochs
        self.print_cost = print_cost

        self.model_ = None
        self.costs_ = None
        self.parameters_ = None
        self._optimizer = None

    def _create_optimizer(self):
        """Create the TensorFlow optimizer based on configuration."""
        if self.optimizer_name == 'gd':
            return tf.keras.optimizers.SGD(learning_rate=self.alpha)
        elif self.optimizer_name == 'momentum':
            return tf.keras.optimizers.SGD(learning_rate=self.alpha, momentum=self.beta)
        elif self.optimizer_name == 'adam':
            return tf.keras.optimizers.Adam(
                learning_rate=self.alpha,
                beta_1=self.beta1,
                beta_2=self.beta2,
                epsilon=self.epsilon
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    @tf.function
    def _train_step(self, x_batch, y_batch):
        """Single training step with explicit gradient computation.

        This method demonstrates the core of backpropagation using GradientTape.
        """
        with tf.GradientTape() as tape:
            # Forward propagation
            predictions = self.model_(x_batch, training=True)
            # Compute loss
            loss = self.model_.compute_loss(y_batch, predictions)

        # Backward propagation - compute gradients
        gradients = tape.gradient(loss, self.model_.trainable_variables)

        # Update weights using optimizer
        self._optimizer.apply_gradients(zip(gradients, self.model_.trainable_variables))

        return loss

    def fit(self, X_train, y_train, seed=None):
        """Train the neural network.

        Parameters
        ----------
        X_train : np.ndarray
            Training data of shape (n_features, m_examples).
        y_train : np.ndarray
            Training labels of shape (1, m_examples).
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        self
        """
        # Configure GPU if available
        configure_gpu_memory_growth()

        # Set random seeds for reproducibility
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # Transpose data to (m_examples, n_features) for TF convention
        X = X_train.T.astype(np.float32)
        y = y_train.T.astype(np.float32)

        m = X.shape[0]  # number of training examples

        # Create model
        self.model_ = BinaryClassifierNN(
            layers_dims=self.layers_dims,
            l2_lambda=self.lambd,
            dropout_rate=self.dropout_rate,
            seed=seed
        )

        # Build model
        self.model_.build(input_shape=(None, X.shape[1]))

        # Create optimizer
        self._optimizer = self._create_optimizer()

        # Training metrics
        self.costs_ = []
        train_loss = tf.keras.metrics.Mean(name='train_loss')

        # Training loop
        for epoch in range(self.num_epochs):
            train_loss.reset_state()

            # Create mini-batches (transpose back for random_mini_batches)
            if seed is not None:
                batch_seed = seed + epoch
            else:
                batch_seed = None
            minibatches = random_mini_batches(X_train, y_train, self.mini_batch_size, batch_seed)

            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch

                # Transpose to (batch_size, features) for TF
                x_batch = tf.constant(minibatch_X.T, dtype=tf.float32)
                y_batch = tf.constant(minibatch_Y.T, dtype=tf.float32)

                # Training step
                loss = self._train_step(x_batch, y_batch)
                train_loss.update_state(loss)

            epoch_cost = train_loss.result().numpy()

            # Print and record cost
            if self.print_cost and epoch % 1000 == 0:
                print(f"Cost after epoch {epoch}: {epoch_cost:.6f}")
            if epoch % 100 == 0:
                self.costs_.append(epoch_cost)

        # Store parameters for compatibility
        self._extract_parameters()

        return self

    def _extract_parameters(self):
        """Extract weights to dictionary format for compatibility."""
        self.parameters_ = {}
        layer_idx = 1

        for layer in self.model_.hidden_layers:
            # Note: weights are stored as (in, out), we need (out, in) for compatibility
            self.parameters_[f'W{layer_idx}'] = layer.w.numpy().T
            self.parameters_[f'b{layer_idx}'] = layer.b.numpy().reshape(-1, 1)
            layer_idx += 1

        # Output layer
        self.parameters_[f'W{layer_idx}'] = self.model_.output_layer.w.numpy().T
        self.parameters_[f'b{layer_idx}'] = self.model_.output_layer.b.numpy().reshape(-1, 1)

    def transform(self, X, y=None):
        """Predict on training data and print accuracy.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_features, m_examples).
        y : np.ndarray, optional
            Labels of shape (1, m_examples).

        Returns
        -------
        np.ndarray
            Predictions of shape (1, m_examples).
        """
        predictions = self._predict_proba(X)
        preds = (predictions > 0.5).astype(int)

        if y is not None:
            accuracy = np.mean(preds == y)
            print(f"Training Accuracy: {accuracy:.6f}")

        return preds

    def predict(self, X, y=None):
        """Predict on test data.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_features, m_examples).
        y : np.ndarray, optional
            Labels of shape (1, m_examples).

        Returns
        -------
        np.ndarray
            Predictions of shape (1, m_examples).
        """
        predictions = self._predict_proba(X)
        preds = (predictions > 0.5).astype(int)

        if y is not None:
            accuracy = np.mean(preds == y)
            print(f"Test Accuracy: {accuracy:.6f}")

        return preds

    def _predict_proba(self, X):
        """Get probability predictions.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_features, m_examples).

        Returns
        -------
        np.ndarray
            Probabilities of shape (1, m_examples).
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Transpose to (m_examples, n_features)
        X_tf = tf.constant(X.T, dtype=tf.float32)
        proba = self.model_(X_tf, training=False).numpy()
        return proba.T

    def predict_proba(self, X):
        """Get probability predictions.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_features, m_examples).

        Returns
        -------
        np.ndarray
            Probabilities of shape (1, m_examples).
        """
        return self._predict_proba(X)

    def plot_costs(self):
        """Plot training cost over epochs."""
        if self.costs_ is None:
            raise ValueError("No training history. Call fit() first.")

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(0, len(self.costs_) * 100, 100), self.costs_)
        plt.ylabel('Cost')
        plt.xlabel('Epoch')
        plt.title(f'Training Cost (lr={self.alpha})')
        plt.grid(True, alpha=0.3)
        plt.show()


def plot_decision_boundary(model, X, y):
    """Plot decision boundary for a binary classifier.

    Parameters
    ----------
    model : callable
        Model with predict method or callable that returns predictions.
    X : np.ndarray
        Data of shape (2, m_examples).
    y : np.ndarray
        Labels of shape (1, m_examples).
    """
    plt.figure(figsize=(10, 8))
    plt.title("Decision Boundary")

    # Set axis limits
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    # Generate grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict on grid
    grid_input = np.c_[xx.ravel(), yy.ravel()].T
    if hasattr(model, 'predict_proba'):
        Z = model.predict_proba(grid_input)
    elif hasattr(model, 'predict'):
        Z = model.predict(grid_input)
    else:
        Z = model(grid_input)

    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.colorbar(label='Probability')

    # Plot data points
    plt.scatter(X[0, :], X[1, :], c=y.flatten(), cmap=plt.cm.Spectral,
                edgecolors='black', linewidth=0.5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
