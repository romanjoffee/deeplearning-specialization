import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from reg_utils import initialize_parameters, load_2D_dataset, plot_decision_boundary, relu, sigmoid
from reg_utils import update_parameters

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def model(X, Y, learning_rate=0.3, num_iterations=30000, keep_prob=1.0, lambd=0.0, print_cost=True):
    grads = {}
    costs = []  # to keep track of the cost
    layers_dims = [X.shape[0], 20, 3, 1]

    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        cost = None

        if lambd == 0 and keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
            cost = compute_cost(a3, Y, parameters)
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0 and keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
            cost = compute_cost(a3, Y, parameters, lambd)
            grads = backward_propagation(X, Y, cache, lambd=lambd)
        elif lambd == 0 and keep_prob < 1:
            a3, cache = forward_propagation(X, parameters, keep_prob=keep_prob)
            cost = compute_cost(a3, Y, parameters)
            grads = backward_propagation(X, Y, cache, keep_prob=keep_prob)
        elif lambd != 0 and keep_prob < 1:
            a3, cache = forward_propagation(X, parameters)
            cost = compute_cost(a3, Y, parameters, lambd)
            grads = backward_propagation(X, Y, cache, lambd=lambd, keep_prob=keep_prob)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    # plot the cost
    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('iterations (x1,000)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return parameters


def forward_propagation(X, parameters, keep_prob=1.0):
    np.random.seed(1)

    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    # dropout
    D1 = np.random.rand(A1.shape[0], A1.shape[1])  # Step 1: initialize matrix D1 from uniform distribution [0, 1)
    D1 = D1 < keep_prob  # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = A1 * D1  # Step 3: shut down some neurons of A1
    A1 = A1 / keep_prob  # Step 4: scale the value of neurons that haven't been shut down

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    # dropout
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


def compute_cost(A, Y, parameters, lambd=0.0):
    m = Y.shape[1]

    # L2 regularization
    reg_parameter = 0.0
    if lambd > 0:
        L = len(parameters) // 2
        for l in range(1, L + 1):
            W = parameters["W" + str(l)]
            reg_parameter += np.sum(np.square(W))

        reg_parameter = lambd / (2 * m) * reg_parameter

    cross_entropy_cost = -1 / m * np.nansum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A)))
    return cross_entropy_cost + reg_parameter


def backward_propagation(X, Y, cache, lambd=0.0, keep_prob=1.0):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T) + lambd / m * W3   # L2 regularization
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2                                      # dropout
    dA2 = dA2 / keep_prob                               # dropout

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T) + lambd / m * W2   # L2 regularization
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1                                      # dropout
    dA1 = dA1 / keep_prob                               # dropout

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T) + lambd / m * W1    # L2 regularization
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def predict(X, y, parameters):
    m = X.shape[1]
    p = np.zeros((1, m), dtype=np.int)

    # Forward propagation
    a3, caches = forward_propagation(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))
    return p


def predict_dec(parameters, X):
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3 > 0.5)
    return predictions


def main():
    train_X, train_Y, test_X, test_Y = load_2D_dataset()

    # base model
    learned_parameters = model(train_X, train_Y)
    print("On the train set:")
    predictions_train = predict(train_X, train_Y, learned_parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, learned_parameters)

    # with L2 regularization
    learned_parameters = model(train_X, train_Y, lambd=0.7)
    print("On the train set:")
    predictions_train = predict(train_X, train_Y, learned_parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, learned_parameters)

    # with dropout
    learned_parameters = model(train_X, train_Y, keep_prob=0.86)
    print("On the train set:")
    predictions_train = predict(train_X, train_Y, learned_parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, learned_parameters)

    # with dropout and L2 regularization
    learned_parameters = model(train_X, train_Y, keep_prob=0.70, lambd=0.52, num_iterations=30000)
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
