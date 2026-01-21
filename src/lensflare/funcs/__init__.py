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

from .tf_funcs import (
    plot_decision_boundary,
    predict_with_model,
    compute_binary_accuracy,
)

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
    # TensorFlow utility functions
    'plot_decision_boundary',
    'np_plot_decision_boundary',
    'predict_with_model',
    'compute_binary_accuracy',
]
