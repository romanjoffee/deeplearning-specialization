import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst.nst_utils import *
import tensorflow as tf


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, tf.transpose(A))
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.reshape(tf.transpose(a_S), (n_C, n_H * n_W))
    a_G = tf.reshape(tf.transpose(a_G), (n_C, n_H * n_W))
    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    # Computing the loss (≈1 line)
    prefix_const = 1 / (4 * (n_H * n_W) ** 2 * n_C ** 2)
    J_style_layer = prefix_const * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

    return J_style_layer


def compute_style_cost(model, STYLE_LAYERS, sess):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, (n_H * n_W, n_C))
    a_G_unrolled = tf.reshape(a_G, (n_H * n_W, n_C))

    # compute the cost with tensorflow (≈1 line)
    J_content = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C, a_G)))

    return J_content


def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style
    return J


def generate(sess, model, input_image, train_step, J_total, J_content, J_style, num_iterations=200):
    # Initialize global variables (you need to run the session on the initializer)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model["input"].assign(input_image))

    generated_image = None

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        ### START CODE HERE ### (1 line)
        sess.run(train_step)
        ### END CODE HERE ###

        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = sess.run(model["input"])
        ### END CODE HERE ###

        # Print every 20 iteration.
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J_total, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('output/generated_image.jpg', generated_image)

    return generated_image


def main():
    # Reset the graph
    tf.reset_default_graph()
    # Start interactive session
    sess = tf.InteractiveSession()

    content_image = scipy.misc.imread("images/louvre_small.jpg")
    content_image = reshape_and_normalize_image(content_image)

    style_image = scipy.misc.imread("images/monet.jpg")
    style_image = reshape_and_normalize_image(style_image)

    input_image = generate_noise_image(content_image)
    imshow(input_image[0])

    model = load_vgg_model("imagenet-vgg-verydeep-19.mat")  # get from S3 deeplearning tarball
    print(model)

    sess.run(model['input'].assign(content_image))

    # Select the output tensor of layer conv4_2
    out = model['conv4_2']

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)

    sess.run(model['input'].assign(style_image))

    # Compute the style cost
    J_style = compute_style_cost(model, STYLE_LAYERS, sess)

    J_total = total_cost(J_content, J_style, alpha=10, beta=40)
    optimizer = tf.train.AdamOptimizer(2.0)

    # define train_step (1 line)
    train_step = optimizer.minimize(J_total)

    # now that we have everything defined, generate image
    generated_image = generate(sess, model, input_image, train_step, J_total, J_content, J_style, num_iterations=200)

    return generated_image


if __name__ == "__main__":
    main()
