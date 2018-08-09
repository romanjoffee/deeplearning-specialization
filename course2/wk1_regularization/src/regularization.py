import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from reg_utils import initialize_parameters, load_2D_dataset, plot_decision_boundary, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, lambd=0.0, keep_prob=1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.

    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """

    grads = {}
    costs = []  # to keep track of the cost
    layers_dims = [X.shape[0], 20, 3, 1]

    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        cost = None

        if lambd == 0 and keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
            cost = compute_cost(a3, Y)
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0 and keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif lambd == 0 and keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
            cost = compute_cost(a3, Y)
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        elif lambd != 0 and keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters)
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            grads = backward_propagation_with_regularization_and_dropout(X, Y, cache, lambd)

        assert (lambd == 0 or keep_prob == 1)  # it is possible to use both L2 regularization and dropout,
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def compute_cost_with_regularization(A, Y, parameters, lambd):
    m = Y.shape[1]
    L = len(parameters) // 2

    reg_parameter = 0
    for l in range(1, L + 1):
        W = parameters["W" + str(l)]
        reg_parameter += np.sum(np.square(W))

    reg_parameter = lambd / (2 * m) * reg_parameter

    cross_entropy_cost = -1 / m * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A)))
    return cross_entropy_cost + reg_parameter


def backward_propagation_with_regularization(X, Y, cache, lambd):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = 1. / m * np.dot(dZ3, A2.T) + W3 * lambd / m
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T) + W2 * lambd / m
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T) + W1 * lambd / m
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def forward_propagation_with_dropout():
    return None
def backward_propagation_with_dropout():
    return None
def backward_propagation_with_regularization_and_dropout():
    return None


def main():
    train_X, train_Y, test_X, test_Y = load_2D_dataset()
    # model(train_X, train_Y)
    learned_parameters = model(train_X, train_Y, lambd=0.7)
    print("On the train set:")
    predictions_train = predict(train_X, train_Y, learned_parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, learned_parameters)
    plt.title("Model with L2-regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(learned_parameters, x.T), train_X, train_Y)


if __name__ == "__main__":
    main()
