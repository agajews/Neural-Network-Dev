import time

num_epochs = 12
h_size = 1024
num_filters = 128
filter_size = 3
pool_size = 2
num_classes = 10
pad = True
img_size = 28
batch_size = 256
learning_rate = 0.001
momentum = 0.99


def main_okapi():
    from OkapiV2.Core import Model, Branch
    from OkapiV2.Layers.Basic import FullyConnected, Dropout
    from OkapiV2.Layers.Activations import ActivationLayer, PReLULayer
    from OkapiV2.Layers.Convolutional import Convolutional, MaxPooling
    from OkapiV2 import Activations, Datasets
    from OkapiV2 import Optimizers
    X_train, y_train, X_val, y_val, X_test, y_test = Datasets.load_mnist()

    tree = Branch()
    tree.add_layer(Convolutional(num_filters, filter_size, filter_size, pad=pad))
    # tree.add_layer(BatchNorm())
    tree.add_layer(PReLULayer())
    tree.add_layer(MaxPooling(pool_size, pool_size))

    tree.add_layer(Convolutional(num_filters, filter_size, filter_size, pad=pad))
    # tree.add_layer(BatchNorm())
    tree.add_layer(PReLULayer())
    tree.add_layer(MaxPooling(pool_size, pool_size))

    tree.add_layer(Convolutional(num_filters, filter_size, filter_size, pad=pad))
    # tree.add_layer(BatchNorm())
    tree.add_layer(PReLULayer())
    tree.add_layer(MaxPooling(pool_size, pool_size))

    '''tree.add_layer(Convolutional(num_filters, filter_size, filter_size, pad=pad))
    # tree.add_layer(BatchNorm())
    tree.add_layer(PReLULayer())
    tree.add_layer(MaxPooling(pool_size, pool_size))'''

    '''tree.add_layer(Dropout(0.25))
    tree.add_layer(FullyConnected((h_size, 1, 1, 1)))
    tree.add_layer(PReLULayer())
    # tree.add_layer(BatchNorm())'''

    tree.add_layer(Dropout(0.5))
    tree.add_layer(FullyConnected())
    tree.add_layer(ActivationLayer(Activations.alt_softmax))

    tree.add_input(X_train)

    model = Model()
    model.set_tree(tree)
    model.set_optimizer(Optimizers.RMSprop(learning_rate=learning_rate,
                                           momentum=momentum))
    model.train([X_train], y_train, num_epochs=num_epochs,
                batch_size=batch_size)
    # ok.save_model(model, 'okapi_mnist.pk')

    okapi_accuracy = model.get_accuracy([X_test], y_test)
    print("Test Accuracy: {}%"
          .format(round(okapi_accuracy, 2)))
    return okapi_accuracy


def main_keras():
    import numpy as np
    # from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    # from keras.utils import np_utils
    from keras.layers.advanced_activations import PReLU
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import RMSprop
    from OkapiV2 import Datasets
    '''(X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, img_size, img_size)
    X_test = X_test.reshape(X_test.shape[0], 1, img_size, img_size)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)'''

    X_train, y_train, X_val, y_val, X_test, y_test = Datasets.load_mnist()
    model = Sequential()

    model.add(Convolution2D(num_filters, filter_size, filter_size,
                            border_mode='same',
                            input_shape=(1, img_size, img_size)))
    # model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Convolution2D(num_filters, filter_size, filter_size,
                            border_mode='same'))
    # model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Convolution2D(num_filters, filter_size, filter_size,
                            border_mode='same'))
    # model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    '''model.add(Convolution2D(num_filters, filter_size, filter_size,
                            border_mode='same'))
    # model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))'''

    '''model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(h_size))
    model.add(PReLU())
    # model.add(BatchNormalization())'''

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    rmsprop = RMSprop(lr=learning_rate, rho=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=num_epochs)
    score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
    keras_accuracy = score[1]
    print('Test accuracy:', keras_accuracy)
    return keras_accuracy

okapi_start = time.clock()
okapi_accuracy = round(main_okapi(), 2)
okapi_time = round(time.clock() - okapi_start, 2)

keras_start = time.clock()
keras_accuracy = round(main_keras() * 100, 2)
keras_time = round(time.clock() - keras_start, 2)

print('Okapi Accuracy: {}, Time: {} \nKeras Accuracy: {}, Time: {}'
      .format(okapi_accuracy, okapi_time, keras_accuracy, keras_time))

