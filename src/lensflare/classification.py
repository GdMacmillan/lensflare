"""Classification models for lensflare."""

from .neural_network.np_models import NpNNClassifier

# Lazy import for TensorFlow classifier
def __getattr__(name):
    if name == 'TfNNClassifier':
        from .neural_network.tf_models import TfNNClassifier
        return TfNNClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['NpNNClassifier', 'TfNNClassifier']
