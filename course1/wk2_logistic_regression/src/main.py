import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from lr_utils import load_dataset


def model(X_train, Y_train, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])

    costs = []

    for i in range(num_iterations):
        A, cost = forward_propagation(w, b, X_train, Y_train)

        gradients = backward_propagation(w, X_train, Y_train, A)

        dw = gradients["dw"]
        db = gradients["db"]

        # Gradient descent
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    learned_params = {"w": w,
                      "b": b}

    return learned_params, costs



def forward_propagation(w, b, X, Y):
    m = X.shape[1]

    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)

    cost = -1 / m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    cost = np.squeeze(cost)

    return A, cost


def forward_propagation(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # compute cost
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return A, cost


def backward_propagation(w, X, Y, A):
    m = X.shape[1]

    dw = 1 / m * np.dot(X, (A - Y).T)  # n X 1
    db = 1 / m * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    gradients = {"dw": dw,
                 "db": db}
    return gradients


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def predict(w, b, X):
    # X - n x m , w - n x 1

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def plot_learning_curve(costs, learning_rate):
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def main():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    train_set_x_flatten_normalized = (X_train_orig.reshape(X_train_orig.shape[0], -1).T) / 255.
    test_set_x_flatten_normalized = (X_test_orig.reshape(X_test_orig.shape[0], -1).T) / 255.

    # hyper-parameters
    num_iterations = 2000
    learning_rate = 0.005

    learned_params, costs = model(train_set_x_flatten_normalized, Y_train_orig, num_iterations, learning_rate, print_cost=True)

    w = learned_params["w"]
    b = learned_params["b"]
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_train = predict(w, b, train_set_x_flatten_normalized)
    Y_prediction_test = predict(w, b, test_set_x_flatten_normalized)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train_orig)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test_orig)) * 100))

    plot_learning_curve(costs, learning_rate)

    result = {"costs": costs,
              "Y_prediction_test": Y_prediction_test,
              "Y_prediction_train": Y_prediction_train,
              "w": w,
              "b": b}
    return result


if __name__ == "__main__":
    main()
