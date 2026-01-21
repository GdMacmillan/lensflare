"""Neural network models for classification."""

from .np_models import NpNNClassifier

from .tf_models import (
    TfNNClassifier,
    BinaryClassifierNN,
    DenseLayer,
    check_gpu_available,
    configure_gpu_memory_growth,
    plot_decision_boundary,
)

__all__ = [
    # NumPy model
    'NpNNClassifier',
    # TensorFlow/Keras models
    'TfNNClassifier',
    'BinaryClassifierNN',
    'DenseLayer',
    'check_gpu_available',
    'configure_gpu_memory_growth',
    'plot_decision_boundary',
]
