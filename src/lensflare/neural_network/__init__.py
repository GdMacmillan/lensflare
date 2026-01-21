"""Neural network models for classification."""

from .np_models import NpNNClassifier

# Lazy imports for TensorFlow models to avoid import errors when TF not configured
_TF_EXPORTS = (
    'TfNNClassifier',
    'BinaryClassifierNN',
    'DenseLayer',
    'check_gpu_available',
    'configure_gpu_memory_growth',
    'plot_decision_boundary',
)

def __getattr__(name):
    if name in _TF_EXPORTS:
        from . import tf_models
        return getattr(tf_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # NumPy model
    'NpNNClassifier',
    # TensorFlow/Keras models (lazy loaded)
    'TfNNClassifier',
    'BinaryClassifierNN',
    'DenseLayer',
    'check_gpu_available',
    'configure_gpu_memory_growth',
    'plot_decision_boundary',
]
