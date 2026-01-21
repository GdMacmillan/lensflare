import numpy as np
from ..funcs import *

class NpNNClassifier(object):
    """Dense Neural Network binary classifier.
    --------------------------------------------------------------------------------
    This class implements a deep neural network entirely in Numpy using an array of
    fully connected layers. Users have a choice of either using gradient-descent,
    momentum or adam optimization. The fully connected layers are made using a relu
    activation function and an optional dropout function after that. Output is a
    linear neuron with a sigmoid activation function

    The parameter layer dims is used to define the depth of layers with the length
    of the array being the total depth and each value stored in the array being the
    number of neurons in that layer.

    All hyperparameters have industry standard defaults and the num of epochs is
    defaulted to 10000. To set a seed for parameter initialization, pass an integer
    to the seed attribute during the fit method call.

    TODO - Understand and fix implementation of dropout so that it works regardless
    optimization method specification (seems to be breaking when Adam specified)

    Parameters
    ----------

    layers_dims : list of ints
        Each of the values in layesr_dims describes the amount of neurons in the
        index i for all i in range(0, len(layers_dims)). If i=0 is an int, then it
        must be equal to X.shape[0] where X is the input training data. The last
        value in this array must be 1 as this network currently only works for
        binary classification.

    optimizer : string, 'gd', 'momentum' or 'adam', optional
        Specifies whether to use gradient descent, momentum or adam optimizer. 'gd'
        by default.

    alpha : float, optional
        Learning rate or the proportion that weights are updated (e.g. 0.001).
        Larger values (e.g. 0.3) results in faster initial learning before the rate
        is updated. Smaller values (e.g. 1.0E-5) slow learning right down during
        training. By default, this is set to 0.0007.

    mini_batch_size : int, optional
        Size of minibatches to use in training. By default, this is set to 64. This
        determines how large of a step to take through the training data before
        updating weights.

    lambd : float, optional
        Describes the regularization strength. If specified, use value specified,
        else is None and l2 regularization not used.

    keep_prob : float, optional
        Dropout criteria for individual neurons. If set, dropout probability will be
        keep_prob value. If not specified, value is None and dropout not used.

    beta : float, optional
        This value refers to the coefficient of the velocity term in classical
        momentum optimization. Part of the controls for how quickly momentum finds
        a minimum of the objective function.

    beta1 : float, optional
        Exponential decay rate for estimates of first moment vector in adam,
        should be in [0, 1). Only used when optimizer='adam'. Default 0.9

    beta2 : float, optional
        Exponential decay rate for estimates of second moment vector in adam,
        should be in [0, 1). Only used when optimizer='adam'. Default 0.999

    epsilon : float, optional
        Very small float value to prevent divide by 0 in implementation of the Adam
        optimization method. By default, this is 1e-8 but a more apropriate number
        may be 1.0 or 0.1 when training an inception network or ImageNet.

    num_epochs : int, optional
        Num of training epochs where each epoch is considered a full pass through
        all training examples. By default the value of this parameters is set to
        10000.

    print_cost : bool, optional
        Optional hyperparemeter to define the verbose printing of the cost at every
        1000 epochs. Deaults to True.

    Attributes
    ----------

    parameters_ : dictionary of arrays
        Each weight array shaped (n_neurons for that layer, n_input features from
        previous layer) Each bias array shaped (n_neurons for that layers, 1). Key
        for array value would be the layer number (e.g. W1 and b1 for the weights
        and bias arrays) W1 would be the weights between input layers(X) and 1st
        hidden layer.

    grads_ : dictionary of arrays
        Each weight grad array shaped (n_neurons for that layer, n_input features
        from previous layer) Each weight bias array shaped (n_neurons for that
        layers, 1). Key for array value would be the layer number (e.g. dW1 and db1
        for the weight and bias grad arrays) dW1 would be the grads for weights with
        respect to cost between input layers(X) and 1st hidden layer.

    costs_ : list of floats
        An array of costs where cost is the objective function output for each
        minibatch step.

    y_train_pred_ : array of ints
        Predicted values of the neural network output layer on training data using
        learned parameters

    y_test_pred_ : array of ints
        Predicted values of the neural network output layer on testing data using
        learned parameters

    Notes
    -----
    This implementation uses Numpy to perform vector operations. It is therefore not
    a parallelized implentation. I implemented this based on the code written in the
    class Improving Deep Neural Networks: Hyperparameter tuning, Regularization and
    Optimization by deeplearning.ai on Coursera. I tried to use an Sklearn style API
    as much as possible for a flexible programming flow designed to function well
    in an exploratory or prototyping setting. Dropout seems to encourage the
    production of NaN values from the cost function. This has not been understood
    and addressed yet so until it is fixed, I encourage you to use my Tensorflow
    implentation.

    --------------------------------------------------------------------------------
    """
    def __init__(self, layers_dims, optimizer="gd", alpha=0.0007, mini_batch_size=64,
                 lambd=None, keep_prob=None, beta=.9, beta1=.9, beta2=.999,
                 epsilon=1e-8, num_epochs=10000, print_cost=True):

        self.layers_dims = layers_dims
        self.optimizer = optimizer
        self.alpha = alpha
        self.mini_batch_size = mini_batch_size
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.num_epochs = num_epochs
        self.print_cost = print_cost
        self.parameters_ = None
        self.grads_ = None
        self.costs_ = None
        self.y_train_pred_ = None
        self.y_test_pred_ = None


    def fit(self, X, y, seed=None):
        # Initialize parameters
        parameters = initialize_parameters(self.layers_dims, seed)
        # Optimize cost function
        self.parameters_, self.grads_, self.costs_ = optimize(parameters,
                                                           X,
                                                           y,
                                                           self.optimizer,
                                                           self.alpha,
                                                           self.mini_batch_size,
                                                           self.lambd,
                                                           self.keep_prob,
                                                           self.beta,
                                                           self.beta1,
                                                           self.beta2,
                                                           self.epsilon,
                                                           self.num_epochs,
                                                           self.print_cost,
                                                           seed)


    def transform(self, X, y=None):

        assert (self.parameters_ is not None)
        self.y_train_pred_ = predict(X, y, self.parameters_)
        return self.y_train_pred_


    def predict(self, X, y=None):

        assert (self.parameters_ is not None)
        self.y_test_pred_ = predict(X, y, self.parameters_, train=False)
        return self.y_test_pred_


    def plot_costs(self):

        assert (self.costs_ is not None)
        # plot the cost
        plt.plot(self.costs_)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(self.alpha))
        plt.show()
