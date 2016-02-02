import theano
import theano.tensor as T
import numpy as np
from Okapi import Layers, Activations, Accuracies, Losses, Initializers, \
    Optimizers
import random
import sys
import time
import pickle


def save_model(model, filename='okapi_model.pk'):
    sys.setrecursionlimit(10000)
    file = open(filename, 'wb')
    pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()


def load_model(filename='okapi_model.pk'):
    file = open(filename, 'rb')
    model = pickle.load(file)
    file.close()
    return model


def tensor_to_four(tensor):
    dims = len(tensor.shape)
    if dims is 4:
        return tensor
    elif dims is 3:
        return np.reshape(tensor, tensor.shape + (1, ))
    elif dims is 2:
        return np.reshape(tensor, tensor.shape + (1, 1, ))
    elif dims is 1:
        return np.reshape(tensor, tensor.shape + (1, 1, 1, ))


class Model():
    def __init__(self, layers=[], batch_size=128):
        self.layers = []
        for layer in layers:
            self.add(layer)
        self.set_batch_size(batch_size)
        self.compiled = False

        def predict_theano(X):
            return X

        self.predict_theano = predict_theano

        self.set_layers_library()
        self.set_activations_library()
        self.set_optimizers_library()
        self.set_initializers_library()

        self.set_train_loss(Losses.CrossentropyTrain())
        self.set_test_loss(Losses.CrossentropyTest())
        self.set_test_accuracy(Accuracies.CatAcc())
        self.set_initializer(Initializers.GlorotUniformInit())
        self.set_optimizer(Optimizers.RMSprop())

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_layers_library(self):
        self.layers_library = [Layers.FullyConnectedLayer,
                               Layers.SimpleRecurrentLayer,
                               Layers.GRULayer,
                               Layers.LSTMLayer,
                               Layers.ActivationLayer,
                               Layers.PReLULayer,
                               Layers.DropoutLayer,
                               Layers.BatchNormalizationLayer,
                               ]

    def add_layers_library(self, layer):
        self.layers_library.append(layer)

    def set_activations_library(self):
        self.activations_library = [Activations.softmax,
                                    Activations.tanh,
                                    Activations.hard_sigmoid,
                                    Activations.sigmoid,
                                    Activations.ReLU]

    def add_activations_library(self, activation):
        self.add_activaitions_library.append(activation)

    def set_initializers_library(self):
        self.initializers_library = [Initializers.NormalInit]

    def add_initializers_library(self, initializer):
        self.initializers_library.append(initializer)

    def set_optimizers_library(self):
        self.optimizers_library = [Optimizers.SGD]

    def add_optimizers_library(self, optimizer):
        self.optimizers_library.append(optimizer)

    def set_rand_model(self, X, y,
                       max_num_layers=2,
                       max_layer_size=10,
                       max_batch_size=128):
        self.layers = []
        num_layers = random.randint(2, max_num_layers)
        for i in range(0, num_layers):
            layer = random.choice(self.layers_library)()
            layer_hyperparams = self.get_rand_hyperparams(
                layer.get_hyperparams_shape(max_layer_size, X.shape))
            layer.set_hyperparams(layer_hyperparams)
            self.layers.append(layer)

        print(self.layers)

        optimizer = random.choice(self.optimizers_library)()
        optimizer_hyperparams = self.get_rand_hyperparams(
            optimizer.get_hyperparams_shape(max_layer_size, X.shape))
        optimizer.set_hyperparams(optimizer_hyperparams)
        self.set_optimizer(optimizer)

        initializer = random.choice(self.initializers_library)()
        initializer_hyperparams = self.get_rand_hyperparams(
            initializer.get_hyperparams_shape(max_layer_size, X.shape))
        initializer.set_hyperparams(initializer_hyperparams)
        self.set_initializer(initializer)

        self.batch_size = random.randint(1, min(max_batch_size, X.shape[0]))

    def get_rand_hyperparams(self, hyperparams_shape):
        if hyperparams_shape is not None:
            hyperparams = []
            for desc in hyperparams_shape:
                if desc[0] == 'int':
                    hyperparams.append(random.randint(desc[1], desc[2]))
                if desc[0] == 'double':
                    hyperparams.append(random.uniform(desc[1], desc[2]))
                if desc[0] == 'boolean':
                    hyperparams.append(bool(random.getrandbits(1)))
                if desc[0] == 'activation':
                    hyperparams.append(random.choice(self.activations_library))
        else:
            hyperparams = None
        return hyperparams

    def add(self, layer):
        self.layers.append(layer)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_train_loss(self, loss):
        self.train_loss = loss

    def set_test_loss(self, loss):
        self.test_loss = loss

    def set_test_accuracy(self, accuracy):
        self.test_accuracy = accuracy

    def set_initializer(self, initializer):
        self.initializer = initializer

    def make_batches(self, X, y, batch_size=None, shuffle=True):
        if batch_size is None:
            batch_size = self.batch_size
        self.num_batches = (y.shape[0] // batch_size) + 1
        if shuffle:
            batch = np.random.permutation(y.shape[0])
            X = X[batch, :, :, :]
            y = y[batch, :, :, :]
        X_batches = np.array_split(X, self.num_batches)
        y_batches = np.array_split(y, self.num_batches)
        return X_batches, y_batches

    def get_param_dims(self, X, y):
        param_dims = []
        prev_output_dim = X.shape
        for layer in self.layers:
            if layer.takes_params:
                param_dims.append(layer.get_param_dims(prev_output_dim))
            else:
                param_dims.append(None)
            prev_output_dim = layer.get_output_dim(prev_output_dim)
        return param_dims

    def get_simple_init_params(self, param_dims):
        rand_params = []
        for param_list in param_dims:
            if param_list is not None:
                init_params = []
                for dim in param_list:
                    init_params.append(
                        self.initializer.get_pre_init_params(dim)
                            .astype('float32'))
            else:
                init_params = None
            rand_params.append(init_params)
        return rand_params

    def get_init_params(self, param_dims, X, y):
        return self.get_simple_init_params(param_dims)

    def compile_theano(self):
        X = T.tensor4(dtype='float32')
        y = T.tensor4(dtype='float32')

        current_layer = X
        for layer, params in zip(self.layers, self.params_shared):
            current_layer = layer.get_output(current_layer, params)

        y_hat = current_layer

        current_layer_test = X
        for layer, params in zip(self.layers, self.params_shared):
            current_layer_test = layer.get_output(current_layer_test,
                                                  params, testing=True)

        y_hat_test = current_layer_test

        train_loss = self.train_loss.get_loss(y_hat, y, self.params_shared)
        test_loss = self.test_loss.get_loss(y_hat_test, y, self.params_shared)
        test_acc = self.test_accuracy.get_accuracy(y_hat_test, y)

        params_optimizer = [p for p in self.params_shared if p is not None]
        params_optimizer = [p for p_list in params_optimizer for p in p_list]

        grads_optimizer = []
        for params in params_optimizer:
            grads_optimizer.append(T.grad(train_loss, params))

        self.train_loss_theano = theano.function([X, y], train_loss)
        self.test_loss_theano = theano.function([X, y], test_loss)
        self.test_acc_theano = theano.function([X, y], test_acc)
        self.predict_theano = theano.function([X], y_hat)
        self.update_step = theano.function(
            inputs=[X, y],
            outputs=train_loss,
            updates=tuple(self.optimizer.get_updates(params_optimizer,
                                                     grads_optimizer,
                                                     train_loss)))

    def set_shared_params(self, init_params):
        self.params_shared = []
        for layer_params in init_params:
            if layer_params is not None:
                layer_params_shared = []
                for params in layer_params:
                    layer_params_shared.append(theano.shared(params))
            else:
                layer_params_shared = None
            self.params_shared.append(layer_params_shared)

    def update_shared_params(self, numpy_updates):
        for shared_layer_params, update_layer_params in \
                zip(self.params_shared, numpy_updates):
            if shared_layer_params is not None:
                for shared_params, update_params in \
                        zip(shared_layer_params, update_layer_params):
                    shared_params.set_value(update_params)

    def set_final_output_shape(self, output_shape):
        for layer in reversed(self.layers):
            if layer.mods_io_dim:
                layer.set_final_output_shape(output_shape)
                return

    def predict(self, input):
        return self.predict_theano(tensor_to_four(input.astype('float32')))

    def get_train_loss(self, X, y):
        if not (len(y.shape) == 4 and len(X.shape) == 4):
            X, y = tensor_to_four(X), tensor_to_four(y)
        return self.train_loss_theano(X, y)

    def get_test_loss(self, X, y):
        if not (len(y.shape) == 4 and len(X.shape) == 4):
            X, y = tensor_to_four(X), tensor_to_four(y)
        return self.test_loss_theano(X, y)

    def get_accuracy(self, X, y):
        if not (len(y.shape) == 4 and len(X.shape) == 4):
            X, y = tensor_to_four(X), tensor_to_four(y)

        X_batches, y_batches = self.make_batches(X, y, self.batch_size)
        accuracy = 0
        for X_batch, y_batch in zip(X_batches, y_batches):
            accuracy += self.test_acc_theano(X_batch, y_batch)
        return accuracy / self.num_batches * 100

    def compile(self, X, y, initialize_params=True):
        self.compiled = True
        print('Compiling model...')
        self.set_final_output_shape(y.shape)
        self.param_dims = self.get_param_dims(X, y)
        if initialize_params:
            self.init_params = self.get_init_params(self.param_dims, X, y)
            self.set_shared_params(self.init_params)
        self.optimizer.setup(self.params_shared)
        self.compile_theano()
        if initialize_params:
            if not self.initializer.is_simple:
                self.init_params = self.initializer.get_init_params(
                    self.param_dims, X, y, self)
            self.update_shared_params(self.init_params)

    def save_params(self, filename='okapi_params.pk'):
        file = open(filename, 'wb')
        pickle.dump(self.params_shared, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load_params(self, filename='okapi_params.pk'):
        file = open(filename, 'rb')
        self.params_shared = pickle.load(file)
        file.close()

    def write_progress(self, epoch, num_epochs, batch_num,
                       time, loss):
        sys.stdout.write(
            "\rEpoch {}/{} | Batch {}/{} | Time: {}s | Loss: {}   "
            .format(epoch + 1, num_epochs,
                    batch_num + 1, self.num_batches,
                    round(time, 1),
                    loss))

    def etr(self, last_time, current_it, total_its):
        its_left = total_its - current_it - 1
        return last_time * its_left

    def train(self, X, y, num_epochs=60, shuffle_each=True, shuffle_start=True,
              batch_callback=None,
              batch_callback_ind=100,
              epoch_callback=None,
              params_filename='okapi_params.pk',
              save_ind=None,
              initialize_params=True):
        X, y = tensor_to_four(X), tensor_to_four(y)
        if not self.compiled:
            self.compile(X, y, initialize_params=initialize_params)
        X_batches, y_batches = self.make_batches(
            X, y, self.batch_size, shuffle_start)
        print('Started training...')
        for epoch in range(0, num_epochs):
            epoch_start = time.clock()
            if shuffle_each:
                X_batches, y_batches = self.make_batches(X, y, self.batch_size)
            total_loss = 0
            for X_batch, y_batch, batch_num in zip(
                    X_batches, y_batches, range(0, self.num_batches)):
                batch_start = time.clock()
                loss = self.update_step(X_batch, y_batch)
                total_loss += loss

                batch_time = time.clock() - batch_start
                time_rem = self.etr(batch_time, batch_num, self.num_batches)
                self.write_progress(epoch, num_epochs, batch_num,
                                    time_rem, loss)

                if batch_callback is not None and \
                        batch_num % batch_callback_ind == 0:
                    batch_callback()

                if save_ind is not None and \
                        batch_num % save_ind == 0:
                    self.save_params(params_filename)

            if save_ind is None:
                self.save_params(params_filename)

            epoch_time = time.clock() - epoch_start
            avg_loss = total_loss / self.num_batches
            self.write_progress(epoch, num_epochs, self.num_batches - 1,
                                epoch_time, avg_loss)

            if epoch_callback is not None:
                epoch_callback()
            print()
