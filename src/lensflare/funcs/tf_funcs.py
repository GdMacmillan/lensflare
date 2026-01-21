import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ..util import random_mini_batches

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    dropout_var -- placeholder for the dropout keep_prob input scalar
    """

    X = tf.placeholder(tf.float32, shape=[n_x, None], name="X")
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name="Y")
    dropout_var = tf.placeholder(tf.float32, name="dropout_var")

    return X, Y, dropout_var

def initialize_parameters(layers_dims, lambd, seed=None):
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

    """

    if seed:
        tf.set_random_seed(seed)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers

    for l in range(1, L + 1):
        # by default lambd is None, if not, lambd is a scalar multiple.
        # lambd = 0.0 disables the regularizer
        if lambd:
            parameters['W' + str(l)] =\
                tf.get_variable('W' + str(l),
                    shape = [layers_dims[l], layers_dims[l - 1]],
                    initializer = tf.contrib.layers.xavier_initializer(seed = seed),
                    regularizer = tf.contrib.layers.l2_regularizer(lambd))

        else:
            parameters['W' + str(l)] =\
                tf.get_variable('W' + str(l),
                    shape = [layers_dims[l], layers_dims[l - 1]],
                    initializer = tf.contrib.layers.xavier_initializer(seed = seed))

        parameters['b' + str(l)] =\
            tf.get_variable('b' + str(l),
                shape = [layers_dims[l], 1],
                initializer = tf.zeros_initializer())

    return parameters

def forward_propagation(X, parameters, dropout_var):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    for l in range(1, L):
        A_prev = A
        Z = tf.add(tf.tensordot(parameters['W' + str(l)], A_prev, 1),
                    parameters['b' + str(l)])
        A = tf.nn.relu(Z)
        A_drop = tf.nn.dropout(A, dropout_var)
    ZL = tf.add(tf.tensordot(parameters['W' + str(L)], A, 1), parameters['b' + str(L)])

    return ZL

def compute_cost(ZL, Y, lambd):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape
        (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for
    # tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits,
                                                                  labels = labels))
    # if lambd then add l2 regularization term to the cost function
    if lambd:
        cost += lambd * np.sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return cost

def predict(x_, y_, train=True):
    m = x_.shape[1]
    preds = np.zeros((1,m), dtype = np.int)

    sess = tf.Session()

    saver = tf.train.import_meta_graph('results/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('results'))

    # Sigmoid transform saved ZL to probas
    X = tf.get_default_graph().get_tensor_by_name("X:0")
    dropout_var = tf.get_default_graph().get_tensor_by_name("dropout_var:0")
    ZL = tf.get_default_graph().get_tensor_by_name("ZL:0")
    AL = tf.nn.sigmoid(ZL)
    probas = sess.run([AL],feed_dict={X:x_, dropout_var:1.0})[0]

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            preds[0,i] = 1
        else:
            preds[0,i] = 0

    # print accuracy if labels provided
    if y_ is not None:
        if train:
            print("Training Accuracy: "  + str(np.mean((preds[0,:] == y_[0,:]))))
        else:
            print("Testing Accuracy: "  + str(np.mean((preds[0,:] == y_[0,:]))))
    sess.close()
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
    Z = model(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z[0].reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.flatten(), cmap=plt.cm.Spectral)
    plt.show()

def predict_dec():
    """
    Used for plotting decision boundary.

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    sess = tf.Session()

    saver = tf.train.import_meta_graph('results/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('results'))

    # Compute sigmoid transformed output on trained weights to compute probas
    X = tf.get_default_graph().get_tensor_by_name("X:0")
    ZL = tf.get_default_graph().get_tensor_by_name("ZL:0")
    dropout_var = tf.get_default_graph().get_tensor_by_name("dropout_var:0")
    AL = tf.nn.sigmoid(ZL)
    # convert probas to predicted binary outputs
    predictions = tf.greater(AL,.5)

    return predictions, X, dropout_var, sess

def optimize(X_train, y_train, X, Y, dropout_var, cost, parameters, optimizer,
             learning_rate, mini_batch_size, keep_prob, beta, beta1, beta2, epsilon,
             num_epochs, print_cost, seed):
    """
    Used to optimize the objective function against data passed in by the fit method.

    Arguments:
    X_train -- Numpy array with training input data, of shape (n features,
        m training examples)
    y_train -- Numpy array with training labels, of shape (1, m training examples)
    X -- training input data tensorflow variable
    Y -- training labels tensorflow variable
    dropout_var -- dropout keep probability tensorflow variable
    cost -- cost tensorflow variable
    parameters -- python dictionary containing network parameter tensorflow variables
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
    costs -- list of costs(cost at each optimization update)
    """
    costs = []                       # to keep track of the cost
    (n_x, m) = X_train.shape
    # Initialize the optimizer
    if optimizer == "gd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate
                                                     ).minimize(cost)
    elif optimizer == "momentum":
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=beta
                                              ).minimize(cost)
    elif optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=beta1,
                                           beta2=beta2,
                                           epsilon=epsilon
                                          ).minimize(cost)

    # Initial a saver to save the trained model
    saver = tf.train.Saver()

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    sess = tf.Session()

    # Run the initialization
    sess.run(init)

    # Do the training loop
    for epoch in range(num_epochs):
        # Defines a cost related to an epoch
        epoch_cost = 0.
        # number of minibatches of size minibatch_size in the train set
        num_minibatches = int(m / mini_batch_size)
        if seed:
            seed = seed + 1
        minibatches = random_mini_batches(X_train, y_train, mini_batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # IMPORTANT: The line that runs the graph on a minibatch.
            _ , minibatch_cost = sess.run([optimizer, cost],
                                          feed_dict={
                                              X: minibatch_X,
                                              Y: minibatch_Y,
                                              dropout_var: keep_prob
                                          })

            epoch_cost += minibatch_cost / num_minibatches

        # Print the cost every 1000th epoch
        if print_cost and epoch % 1000 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost and epoch % 100 == 0:
            costs.append(epoch_cost)

    saver.save(sess, 'results/model')
    parameters = sess.run(parameters)

    sess.close()
    return parameters, costs
