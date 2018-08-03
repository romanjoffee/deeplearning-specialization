import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)


def nn_model(X, Y, n_h, num_iterations, learning_rate, print_cost=False):
    n_x, n_y = layer_sizes(X, Y)

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        cache = forward_propagation(X, parameters)

        gradients = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, gradients, learning_rate)

        if print_cost and i % 1000 == 0:
            Yhat = cache["A2"]
            cost = compute_cost(Yhat, Y)
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return cache


def backward_propagation(parameters, cache, X, Y):
    """
    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    Returns:
    grads -- dictionary containing gradients with respect to different parameters
    """

    m = X.shape[1]

    W2 = parameters["W2"]
    A1 = cache["A1"]
    Yhat = cache["A2"]

    dZ2 = Yhat - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using the gradient descent update `Theta := Theta - 1/alpha * dTheta`
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def compute_cost(Yhat, Y):
    """
    Computes the cross-entropy cost
    """
    cross_entropy_loss = np.multiply(Y, np.log(Yhat)) + np.multiply((1 - Y), np.log(1 - Yhat))

    m = Y.shape[1]  # number of examples
    cost = -1 / m * np.sum(cross_entropy_loss)

    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect. # E.g., turns [[17]] into 17
    assert (isinstance(cost, float))

    return cost


def layer_sizes(X, Y):
    n_x = X.shape[0]  # size of input layer, X of size `n (features) x m (examples)`
    n_y = Y.shape[0]  # size of output layer Y of size `1 x m (examples)`
    return n_x, n_y


def predict(parameters, X):
    cache = forward_propagation(X, parameters)
    Yhat = cache["A2"]
    predictions = (Yhat > 0.5)
    return predictions


def main():
    X, Y = load_planar_dataset()

    #hyper-parameters
    num_hidden_units = 4
    num_iterations = 10000
    learning_rate = 1.2

    learner_parameters = nn_model(X, Y, num_hidden_units, num_iterations, learning_rate, print_cost=True)

    predictions = predict(learner_parameters, X)
    print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    plot_decision_boundary(lambda x: predict(learner_parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()


if __name__ == "__main__":
    main()