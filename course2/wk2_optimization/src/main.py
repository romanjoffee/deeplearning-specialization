import gradientdescent
import momentum
import adam

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    """
    3-layer neural network model which can be run in different optimizer modes.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    v = None
    s = None
    cost = None
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update
    seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = momentum.initialize(parameters)
    elif optimizer == "adam":
        v, s = adam.initialize(parameters)

    # Optimization loop
    for i in range(num_epochs):
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = gradientdescent.random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost
            cost = compute_cost(a3, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = gradientdescent.update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = momentum.update_parameters(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = adam.update_parameters(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2, epsilon)

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


def main():
    train_X, train_Y = load_dataset()
    layers_dims = [train_X.shape[0], 5, 2, 1]
    print("training with mini-batch gd optimizer")
    learned_parameters_gd = model(train_X, train_Y, layers_dims, optimizer="gd")
    predict(train_X, train_Y, learned_parameters_gd)
    # Plot decision boundary
    plt.title("model with Gradient Descent optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: predict_dec(learned_parameters_gd, x.T), train_X, train_Y)

    print("training with momentum optimizer")
    learned_parameters_momentum = model(train_X, train_Y, layers_dims, beta=0.9, optimizer="momentum")
    predict(train_X, train_Y, learned_parameters_momentum)
    # Plot decision boundary
    plt.title("model with Momentum optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: predict_dec(learned_parameters_momentum, x.T), train_X, train_Y)

    print("training with adam optimizer")
    learned_parameters_adam = model(train_X, train_Y, layers_dims, optimizer="adam")
    predict(train_X, train_Y, learned_parameters_adam)
    # Plot decision boundary
    plt.title("model with Adam optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: predict_dec(learned_parameters_adam, x.T), train_X, train_Y)

    return None


if __name__ == "__main__":
    main()
