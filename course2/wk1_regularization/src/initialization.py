import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he"):
    costs = []  # to keep track of the loss
    layers_dims = [X.shape[0], 10, 5, 1]

    parameters = {}
    if initialization == "random":
        parameters = bad_initialization_relu_act_fn(layers_dims)
    elif initialization == "he":
        parameters = good_initialization_relu_act_fn(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)
        # Loss
        cost = compute_loss(a3, Y)
        # Backward propagation.
        grads = backward_propagation(X, Y, cache)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def good_initialization_relu_act_fn(layers_dim):
    L = len(layers_dim)
    params = {}
    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layers_dim[l], layers_dim[l - 1]) * np.sqrt(2 / layers_dim[l - 1])
        params["b" + str(l)] = np.zeros((layers_dim[l], 1))

    return params


def bad_initialization_relu_act_fn(layers_dim):
    np.random.seed(3)

    L = len(layers_dim)
    params = {}
    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layers_dim[l], layers_dim[l - 1]) * 10
        params["b" + str(l)] = np.zeros((layers_dim[l], 1))

    return params


def main():
    train_X, train_Y, test_X, test_Y = load_dataset()

    # not so good initialization
    parameters = model(train_X, train_Y, initialization="random")
    print("On the train set:")
    predict(train_X, train_Y, parameters)
    print("On the test set:")
    predict(test_X, test_Y, parameters)

    # good initialization
    parameters = model(train_X, train_Y, initialization="he")
    print("On the train set:")
    predict(train_X, train_Y, parameters)
    print("On the test set:")
    predict(test_X, test_Y, parameters)
    plt.title("Model with He initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


if __name__ == "__main__":
    main()
