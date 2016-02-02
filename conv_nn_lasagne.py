import numpy as np
import theano
import theano.tensor as T
import lasagne
from urllib.request import urlretrieve
import gzip
import os
import time


def load_dataset():
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn(input_var, depth=2, filter_size=5, pool_size=2, dropout_rate=0.5,
              nonlinearity=lasagne.nonlinearities.rectify,
              num_filters=32, final_connected_size=256):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=num_filters,
            filter_size=(filter_size, filter_size),
            nonlinearity=nonlinearity,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network,
                                            pool_size=(pool_size,
                                                       pool_size))

    for i in range(0, depth-1):
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=num_filters,
                filter_size=(filter_size, filter_size),
                nonlinearity=nonlinearity)
        network = lasagne.layers.MaxPool2DLayer(network,
                                                pool_size=(pool_size,
                                                           pool_size))

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=dropout_rate),
            num_units=final_connected_size,
            nonlinearity=nonlinearity)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=dropout_rate),
            num_units=final_connected_size,
            nonlinearity=nonlinearity)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=dropout_rate),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(depth=2, filter_size=5, pool_size=2, dropout_rate=0.5,
         nonlinearity=lasagne.nonlinearities.rectify,
         num_filters=32, final_connected_size=256,
         learning_rate=0.005, avg_decay=0.999, batch_size=256,
         num_epochs=500):

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var, depth=depth, filter_size=filter_size,
                        pool_size=pool_size,
                        dropout_rate=dropout_rate,
                        nonlinearity=nonlinearity,
                        num_filters=num_filters,
                        final_connected_size=final_connected_size)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)\
        .mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.rmsprop(loss, params,
                                      learning_rate=learning_rate,
                                      rho=avg_decay,
                                      epsilon=1e-08)

    '''updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)'''

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var).mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("Starting training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size,
                                         shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size,
                                         shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    print("Saving trained parameters...")
    np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    print("Done")

main(batch_size=500)
