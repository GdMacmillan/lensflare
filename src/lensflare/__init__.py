# -*- encoding: utf-8 -*-
"""LensFlare - Educational deep learning library.

This library provides educational implementations of neural networks using
both NumPy and TensorFlow 2 with the low-level GradientTape API.

On Apple Silicon Macs, install tensorflow-metal for GPU acceleration:
    pip install tensorflow-metal
"""

import sys

from .__version__ import __version__

# Check Python version
if sys.version_info < (3, 12):
    raise ValueError(
        f'Unsupported Python version {sys.version_info.major}.{sys.version_info.minor}. '
        'LensFlare requires Python 3.12 or higher.'
    )

# Public API - NumPy components (always available)
from .neural_network import NpNNClassifier
from .util import load_moons_dataset, random_mini_batches

# Lazy imports for TensorFlow components
_TF_EXPORTS = (
    'TfNNClassifier',
    'check_gpu_available',
    'configure_gpu_memory_growth',
    'BinaryClassifierNN',
    'DenseLayer',
    'plot_decision_boundary',
)

def __getattr__(name):
    if name in _TF_EXPORTS:
        if name == 'TfNNClassifier':
            from .neural_network import TfNNClassifier
            return TfNNClassifier
        elif name == 'plot_decision_boundary':
            from .funcs import plot_decision_boundary
            return plot_decision_boundary
        else:
            from . import neural_network
            return getattr(neural_network, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    '__version__',
    # Classifiers
    'TfNNClassifier',
    'NpNNClassifier',
    # Utilities
    'load_moons_dataset',
    'random_mini_batches',
    'plot_decision_boundary',
    # GPU utilities
    'check_gpu_available',
    'configure_gpu_memory_growth',
    # Model building blocks
    'BinaryClassifierNN',
    'DenseLayer',
]
