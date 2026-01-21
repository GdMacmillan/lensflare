"""Neural network functions for forward/backward propagation and optimization."""

from .np_funcs import (
    sigmoid,
    relu,
    relu_backward,
    sigmoid_backward,
    initialize_parameters,
    initialize_velocity,
    initialize_adam,
    linear_forward,
    linear_activation_forward,
    forward_propagation,
    compute_cost,
    linear_backward,
    linear_activation_backward,
    backward_propagation,
    update_parameters_with_gd,
    update_parameters_with_momentum,
    update_parameters_with_adam,
    predict,
    plot_decision_boundary as np_plot_decision_boundary,
    predict_dec,
    optimize,
)

# Lazy import for TensorFlow functions to avoid import errors when TF not configured
def __getattr__(name):
    if name in ('plot_decision_boundary', 'predict_with_model', 'compute_binary_accuracy'):
        from . import tf_funcs
        return getattr(tf_funcs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # NumPy functions
    'sigmoid',
    'relu',
    'relu_backward',
    'sigmoid_backward',
    'initialize_parameters',
    'initialize_velocity',
    'initialize_adam',
    'linear_forward',
    'linear_activation_forward',
    'forward_propagation',
    'compute_cost',
    'linear_backward',
    'linear_activation_backward',
    'backward_propagation',
    'update_parameters_with_gd',
    'update_parameters_with_momentum',
    'update_parameters_with_adam',
    'predict',
    'predict_dec',
    'optimize',
    # TensorFlow utility functions (lazy loaded)
    'plot_decision_boundary',
    'np_plot_decision_boundary',
    'predict_with_model',
    'compute_binary_accuracy',
]
