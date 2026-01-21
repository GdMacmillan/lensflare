"""TensorFlow neural network models.

This module provides the TfNNClassifier for binary classification using
TensorFlow 2 with the low-level GradientTape API for educational purposes.

The implementation has been modernized to use TF2/Keras while maintaining
the original sklearn-style API.
"""

# Re-export from keras_models for backwards compatibility
from .keras_models import (
    TfNNClassifier,
    BinaryClassifierNN,
    DenseLayer,
    check_gpu_available,
    configure_gpu_memory_growth,
    plot_decision_boundary,
)

__all__ = [
    'TfNNClassifier',
    'BinaryClassifierNN',
    'DenseLayer',
    'check_gpu_available',
    'configure_gpu_memory_growth',
    'plot_decision_boundary',
]
