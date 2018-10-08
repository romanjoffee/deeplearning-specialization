import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

import matplotlib as mpl
mpl.use('TkAgg')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *


def model_graph(model):
    plot_model(model, to_file='model.png')
    SVG(model_to_dot(model).create(prog='dot', format='svg'))


def model_Sequential(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32,
                                  kernel_size=7,
                                  strides=(1, 1),
                                  padding='same',
                                  activation='relu',
                                  input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=2))

    # model.add(keras.layers.Dropout(0.9))
    # model.add(keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    # model.add(keras.layers.MaxPooling2D(pool_size=2))
    # model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model


def model_Def(input_shape):
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='model')
    return model


def main():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Normalize image vectors
    x_train_normalized = X_train_orig / 255.
    x_test_normalized = X_test_orig / 255.

    # Reshape
    y_train_normalized = Y_train_orig.T
    y_test_normalized = Y_test_orig.T

    (x_train, x_valid) = x_train_normalized[50:], x_train_normalized[:50]
    (y_train, y_valid) = y_train_normalized[50:], y_train_normalized[:50]

    checkpoint = keras.callbacks.ModelCheckpoint(filepath="model.weights.best.hdf5", verbose=True, save_best_only=True)

    model = model_Sequential((x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model.fit(x_train, y_train, batch_size=64, epochs=30,
    #           validation_data=(x_valid, y_valid),
    #           callbacks=[checkpoint], verbose=1)

    model_graph(model)

    model.load_weights('model.weights.best.hdf5')

    preds = model.evaluate(x_test_normalized, y_test_normalized)
    print()
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    img_path = 'images/happy.jpeg'
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(model.predict(x))


if __name__ == "__main__":
    main()
