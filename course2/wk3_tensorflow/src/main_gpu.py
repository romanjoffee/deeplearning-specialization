import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

import main_cpu as m


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=500, minibatch_size=64, print_cost=True):
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    costs = []  # To keep track of the cost

    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")

    with tf.device("/gpu:0"):
        parameters = m.initialize_parameters()
        # Forward propagation: Build the forward propagation in the tensorflow graph
        forward_prop_def = m.forward_propagation(X, parameters)
        # Cost function: Add cost function to tensorflow graph
        cost_def = m.compute_cost(forward_prop_def, Y)
        # Backward propagation definition: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_def)

    # Initialize all the variables
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True) #, log_device_placement=True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # Run the initialization
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = sess.run([optimizer, cost_def], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        with tf.device("/cpu:0"):
            parameters = sess.run(parameters)
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(forward_prop_def), tf.argmax(Y))
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        return parameters

def main():

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten / 255.
    X_test = X_test_flatten / 255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)
    #
    start_time = time.time()
    learned_parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=600)

    print("--- %s seconds ---" % (time.time() - start_time))

    return learned_parameters


if __name__=="__main__":
    main()