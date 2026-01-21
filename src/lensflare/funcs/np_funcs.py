import math
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z, leaky=False):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward
        pass efficiently
    """

    if leaky:
        A = np.maximum(Z * 0.01, Z)
    else:
        A = np.maximum(0,Z)

    assert(A.shape == Z.shape)

    cache = Z
    return A, cache

def relu_backward(dA, cache, leaky=False):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    if leaky:
        dZ[Z <= 0] = 0.01
    else:
        dZ[Z <= 0] = 0.0

    assert (dZ.shape == Z.shape)

    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ

def initialize_parameters(layer_dims, seed):
    """
    Initializes parameters using He initialization.
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our
        network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ...,
        "WL", "bL":
                    W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    b1 -- bias vector of shape (layer_dims[l], 1)
                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    bl -- bias vector of shape (1, layer_dims[l])

    Tips:
    - For example: the layer_dims if are [2,2,1], this means W1's shape was (2,2),
    b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
    - In for loops, use parameters['W' + str(l)] to access Wl, where l is the
    iterative integer.
    """

    if seed:
        np.random.seed(seed)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers

    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],
                                                   layers_dims[l - 1]) *\
                                       np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the
                    corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl

    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}

    # Initialize velocity
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v

def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the
                    corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns:
    v -- python dictionary that will contain the exponentially weighted average of
        the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of
        the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v, s

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer,
        number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of
        previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing
        the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Forward propagation for the LINEAR->ACTIVATION layer
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous
        layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of
        previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string:
        "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def forward_propagation(X, parameters, keep_prob):
    """
    Forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed
                from 0 to L-2) the cache of linear_sigmoid_forward() (there is one,
                indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                # number of layers in the neural network

    # [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             parameters['W' + str(l)],
                                             parameters['b' + str(l)],
                                             activation='relu')
        if keep_prob:
            D = np.random.rand(*A.shape)
            D = D < keep_prob
            A = A * D
            A = A / keep_prob
            linear_cache, activation_cache = cache
            new_linear_cache = linear_cache + (D,)
            cache = (new_linear_cache, activation_cache)

        caches.append(cache)
    # LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A,
                                          parameters['W' + str(L)],
                                          parameters['b' + str(L)],
                                          activation='sigmoid')
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))

    return AL, caches

def compute_cost(AL, Y, parameters, lambd):

    """
    Implements the cost function
    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3

    Returns:
    cost - value of the cost function
    """

    L = len(parameters) // 2
    m = Y.shape[1]
    # cross entropy cost
    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost = 1. / m * np.sum(logprobs)

    if lambd:
        L2_regularization_cost = lambd / (2. * m) * np.sum([np.sum(np.square(parameters["W" + str(l)])) for l in range(1, L + 1)])
        cost += L2_regularization_cost
    return cost

def linear_backward(dZ, cache, lambd):
    """
    Linear portion of backward propagation for a single layer (layer l)
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in
        the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous
        layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    if len(cache) == 4:
        A_prev, W, b, D = cache
    else:
        A_prev, W, b = cache
    m = A_prev.shape[1]

    if lambd:
        dW = 1. / m * np.dot(dZ, A_prev.T) + lambd / m * W
    else:
        dW = 1. / m * np.dot(dZ, A_prev.T)

    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, lambd, keep_prob, activation):
    """
    Backward propagation for the LINEAR->ACTIVATION layer.
    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for
        computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string:
        "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous
        layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    linear_cache, activation_cache = cache

    if len(linear_cache) == 4:
        A, W, b, D = linear_cache
        dA = dA * D
        dA = dA / keep_prob

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    return dA_prev, dW, db

def backward_propagation(AL, Y, caches, lambd, keep_prob):
    """
    Backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's
                caches[l], for l in range(L-1) i.e l = 0...L-2) the cache of
                linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients.
    current_cache = caches[L - 1]
    grads["dA" + str(L)],\
    grads["dW" + str(L)],\
    grads["db" + str(L)] = linear_activation_backward(dAL,
                                                      current_cache,
                                                      lambd,
                                                      keep_prob,
                                                      activation='sigmoid')
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp,\
        dW_temp,\
        db_temp = linear_activation_backward(grads['dA' + str(l + 2)],
                                             current_cache,
                                             lambd,
                                             keep_prob,
                                             activation='relu')
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    L = len(parameters) // 2 # number of layers in the neural networks

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads['dW' + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads['db' + str(l + 1)]

    return parameters

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2 # number of layers in the neural networks

    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] +\
                   (1 - beta) * grads['dW' + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] +\
                   (1 - beta) * grads['db' + str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] -= learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * v["db" + str(l+1)]

    return parameters, v

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2    # number of layers in the neural networks
    v_corrected = {}            # Initializing first moment estimate, python dictionary
    s_corrected = {}            # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients.
        v["dW" + str(l + 1)] = np.multiply(beta1, v["dW" + str(l + 1)]) +\
                               np.multiply((1 - beta1), grads["dW" + str(l + 1)])
        v["db" + str(l + 1)] = np.multiply(beta1, v["db" + str(l + 1)]) +\
                               np.multiply((1 - beta1), grads["db" + str(l + 1)])

        # Compute bias-corrected first moment estimate.
        v_corrected["dW" + str(l + 1)] = np.divide(v["dW" + str(l + 1)], 1 - beta1 ** t)
        v_corrected["db" + str(l + 1)] = np.divide(v["db" + str(l + 1)], 1 - beta1 ** t)

        # Moving average of the squared gradients.
        s["dW" + str(l + 1)] = np.multiply(beta2, s["dW" + str(l + 1)]) +\
                               np.multiply((1 - beta2), grads["dW" + str(l + 1)] ** 2)
        s["db" + str(l + 1)] = np.multiply(beta2, s["db" + str(l + 1)]) +\
                               np.multiply((1 - beta2), grads["db" + str(l + 1)] ** 2)

        # Compute bias-corrected second raw moment estimate.
        s_corrected["dW" + str(l + 1)] = np.divide(s["dW" + str(l + 1)], 1 - beta2 ** t)
        s_corrected["db" + str(l + 1)] = np.divide(s["db" + str(l + 1)], 1 - beta2 ** t)

        # Update parameters.
        parameters["W" + str(l + 1)] -= learning_rate *\
                                np.divide(v_corrected["dW" + str(l + 1)],
                                    np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon)
        parameters["b" + str(l + 1)] -= learning_rate *\
                                np.divide(v_corrected["db" + str(l + 1)],
                                    np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)

    return parameters, v, s

def predict(X, Y, parameters, train=True):
    """
    This function is used to predict the results of a  n-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    preds = np.zeros((1,m), dtype = np.int)

    # Forward propagation
    keep_prob=None
    AL, caches = forward_propagation(X, parameters, keep_prob)

    # convert probas to 0/1 predictions
    for i in range(0, AL.shape[1]):
        if AL[0,i] > 0.5:
            preds[0,i] = 1
        else:
            preds[0,i] = 0

    # print accuracy if labels provided
    if Y is not None:
        if train:
            print("Training Accuracy: "  + str(np.mean((preds[0,:] == Y[0,:]))))
        else:
            print("Testing Accuracy: "  + str(np.mean((preds[0,:] == Y[0,:]))))

    return preds

def plot_decision_boundary(model, X, y):

    plt.title("Model with Gradient Descent optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])

    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()

def predict_dec(clf, X):
    """
    Used for plotting decision boundary.

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5
    parameters = clf.parameters_
    keep_prob = None
    AL, cache = forward_propagation(X, parameters, keep_prob)
    predictions = (AL > 0.5)
    return predictions

def optimize(parameters, X, Y, optimizer, learning_rate, mini_batch_size, lambd,
             keep_prob, beta, beta1, beta2, epsilon, num_epochs, print_cost, seed):
    """
    Used to optimize the objective function against data passed in by the fit method.

    Arguments:
    parameters -- python dictionary containing network parameters
    X -- training inputs, of shape (n features, m training examples)
    Y -- training labels, of shape (1, m training examples)
    optimizer -- name of optimization algorithm to use
    learning_rate -- learning rate of the optimization
    mini_batch_size -- size of minibatch to perform gradient update steps
    lambd -- l2 regularization parameter
    keep_prob -- dropout regularization parameter
    beta -- momentum optimization parameter
    beta1 -- adam optimization parameter
    beta2 -- adam optimization parameter
    epsilon -- small error term to prevent divide by zero in adam
    num_epochs -- number of epochs of the optimization loop
    print_cost -- boolean to set whether costs are printed

    Returns:
    parameters -- python dictionary of updated weights
    grads -- python dictionary of gradients used to compute weight updates
    costs -- list of costs(cost at each optimization update)
    """
    # to keep track of the cost
    costs = []
    # initializing the counter required for Adam update
    t = 0

    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):

        if seed:
            seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            AL, caches = forward_propagation(minibatch_X, parameters, keep_prob)

            # Compute cost
            cost = compute_cost(AL, minibatch_Y, parameters, lambd)

            # Backward propagation
            grads = backward_propagation(AL, minibatch_Y, caches, lambd, keep_prob)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters,
                                                       grads,
                                                       learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters,
                                                                grads,
                                                                v,
                                                                beta,
                                                                learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters,
                                                               grads,
                                                               v,
                                                               s,
                                                               t,
                                                               learning_rate,
                                                               beta1,
                                                               beta2,
                                                               epsilon)

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    return parameters, grads, costs
