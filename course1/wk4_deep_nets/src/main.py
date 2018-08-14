import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import dnn_utils as utils

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


#start forward pass
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    Yhat, cache = linear_activation_forward(A, W, b, "sigmoid")
    caches.append(cache)

    assert (Yhat.shape == (1, X.shape[1]))

    return Yhat, caches


def linear_activation_forward(A_prev, W, b, activationFn):
    """
    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "A_prev", "W", "b" and corresponding "Z"; stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A_prev) + b
    assert (Z.shape == (W.shape[0], A_prev.shape[1]))
    cache = (A_prev, W, b, Z)

    if activationFn == "relu":
        A, activation_cache = utils.relu(Z)
    else:
        A, activation_cache = utils.sigmoid(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    return A, cache

#end forward pass


# start backward pass
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # dL/dA, derivative of cross-entropy loss function

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]   #for the l-th layer since that collection is 0-based
    grads["dW" + str(L)], grads["db" + str(L)], grads["dA" + str(L - 1)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    # Loop from l=L-1 to l=1
    for l in reversed(range(1, L)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "dL_dA, current_cache". Outputs: "dL_dW , dL_db, dL_dA_prev
        current_cache = caches[l - 1]
        dL_dA = grads["dA" + str(l)]
        dL_dW, dL_db, dA_prev = linear_activation_backward(dL_dA, current_cache, "relu")
        grads["dW" + str(l)] = dL_dW
        grads["db" + str(l)] = dL_db
        grads["dA" + str(l - 1)] = dA_prev      # think of this step as -> dA[l-1] = W[l] * dL_dZ[l]

    return grads


def linear_activation_backward(dL_dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b, Z = cache

    dL_dZ = None
    if activation == "relu":
        dL_dZ = utils.relu_backward(dL_dA, Z)
    elif activation == "sigmoid":
        dL_dZ = utils.sigmoid_backward(dL_dA, Z)

    m = A_prev.shape[1]

    dL_dW = 1 / m * np.dot(dL_dZ, A_prev.T)                 # dL_dA * dA_dZ * dZ_dW = dL_dW
    dL_db = 1 / m * np.sum(dL_dZ, axis=1, keepdims=True)    # dL_dA * dA_dZ * dZ_db = dL_db

    dA_prev = np.dot(W.T, dL_dZ)                            # think of this step as -> dA[l-1] = W[l] * dL_dZ[l]

    assert (dA_prev.shape == A_prev.shape)
    assert (dL_dW.shape == W.shape)
    assert (dL_db.shape == b.shape)

    return dL_dW, dL_db, dA_prev

#end backward pass


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    p = np.zeros((1, m))

    # Forward propagation
    Yhat, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, Yhat.shape[1]):
        if Yhat[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p


def print_mislabeled_images(classes, X, y, p):
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))
    plt.show()


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = utils.initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        Yhat, caches = L_model_forward(X, parameters)

        # Backward propagation
        gradients = L_model_backward(Yhat, Y, caches)

        # Update parameters
        parameters = update_parameters(parameters, gradients, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            cost = utils.compute_cost(Yhat, Y)
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = utils.load_data()
    train_x_flatten_normalized = train_x_orig.reshape(train_x_orig.shape[0],
                                                      -1).T / 255.  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten_normalized = test_x_orig.reshape(test_x_orig.shape[0], -1).T / 255.

    # hyper-parameters
    layers = [64 * 64 * 3, 20, 7, 5, 1]  # 64 by 64 image, 3 layers for RGB, hence X vector has 64 x 64 x 3 features
    num_iterations = 2500
    learning_rate = 0.009

    learner_parameters = L_layer_model(train_x_flatten_normalized, train_y, layers, learning_rate, num_iterations, True)

    predict(train_x_flatten_normalized, train_y, learner_parameters)
    pred_test = predict(test_x_flatten_normalized, test_y, learner_parameters)

    print_mislabeled_images(classes, test_x_flatten_normalized, test_y, pred_test)


if __name__ == "__main__":
    main()