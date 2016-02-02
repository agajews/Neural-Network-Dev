from OkapiV2 import Losses, Accuracies, Optimizers, Initializers, Activations
import theano
import theano.tensor as T
import numpy as np
import sys
import pickle
import time


def atleast_4d(x):
    if x.ndim < 4:
        return np.expand_dims(np.atleast_3d(x), axis=3).astype('float32')
    else:
        return x.astype('float32')


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


def make_batches(X, y, batch_size=128, shuffle=True):
    X, y = atleast_4d(X), atleast_4d(y)
    num_batches = (y.shape[0] // batch_size) + 1
    if shuffle:
        batch = np.random.permutation(y.shape[0])
        X = X[batch, :, :, :]
        y = y[batch, :, :, :]
    X_batches = np.array_split(X, num_batches)
    y_batches = np.array_split(y, num_batches)
    return X_batches, y_batches, num_batches


class Branch():
    def __init__(self):
        self.inputs = []
        self.merge_mode = 'flat_append'
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_input(self, input):
        self.inputs.append(input)

    def get_output(self, inputs, params_list, testing=False):
        for i in range(len(inputs)):
            if isinstance(inputs[i], np.ndarray):
                inputs[i] = T.tensor4(dtype='float32')
            if self.merge_mode is 'flat_append':
                inputs[i] = inputs[i].flatten(1)
            else:
                raise Exception('Invalid merge mode')
        X = T.concatenate(inputs)
        current_layer = X
        for layer, params in zip(self.layers, params_list):
            current_layer = layer.get_output(current_layer, params, testing)
        output = current_layer
        return output


class Model():
    def __init__(self):
        self.compiled = False
        self.dream_compiled = False
        self.layers = []
        self.set_loss(Losses.Crossentropy())
        self.set_accuracy(Accuracies.Categorical())
        self.set_optimizer(Optimizers.RMSprop())

    def save_params(self, filename='okapi_params.pk'):
        file = open(filename, 'wb')
        pickle.dump(self.params_shared, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load_params(self, filename='okapi_params.pk'):
        file = open(filename, 'rb')
        self.params_shared = pickle.load(file)
        file.close()

    def add(self, layer):
        self.layers.append(layer)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss(self, loss):
        self.loss = loss

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    def get_init_params(self, X, y):
        prev_output_dim = X.shape
        init_params = []
        for layer in self.layers:
            output_dim = layer.get_output_dim(prev_output_dim)
            layer_inits = layer.get_init_params(prev_output_dim)
            if layer_inits is not None:
                layer_inits = [i.astype('float32') for i in layer_inits]
            init_params.append(layer_inits)
            prev_output_dim = output_dim
        return init_params

    def initialize_params(self, X, y):
        self.init_params = self.get_init_params(X, y)
        self.params_shared = []
        for layer_params in self.init_params:
            if layer_params is not None:
                layer_params_shared = []
                for params in layer_params:
                    layer_params_shared.append(theano.shared(params))
            else:
                layer_params_shared = None
            self.params_shared.append(layer_params_shared)

    def randomize_params(self, X, y):
        rand_params = self.get_init_params(X, y)
        for current_params, new_params in zip(self.params_shared, rand_params):
            if current_params is not None:
                for current, rand in zip(current_params, new_params):
                    current.set_value(rand)

    def set_params_as_vec(self, params):
        index = 0
        params = params.astype('float32')
        for current_params in self.params_shared:
            if current_params is not None:
                index = 0
        params = params.astype('float32')
        for current_params in self.params_shared:
            if current_params is not None:
                for current in current_params:
                    shape = current.get_value().shape
                    size = np.prod(shape)
                    new_params = params[index:index + size].reshape(shape)
                    current.set_value(new_params)
                    index += size

    def get_params_as_vec(self):
        vec = []
        for current_params in self.params_shared:
            if current_params is not None:
                for current in current_params:
                    vec = np.append(vec, current.get_value().flat)
        return vec

    def compile(self, X_train, y_train, initialize_params=True):
        print('Compiling model...')
        self.compiled = True
        try:
            self.num_output_dims
        except:
            self.num_output_dims = y_train.ndim
        X_train, y_train = atleast_4d(X_train), atleast_4d(y_train)
        self.set_final_output_shape(y_train.shape)
        if initialize_params:
            self.initialize_params(X_train, y_train)

        self.optimizer.build(self.init_params)

        X = T.tensor4(dtype='float32')
        y = T.tensor4(dtype='float32')

        current_layer = X
        for layer, params in zip(self.layers, self.params_shared):
            current_layer = layer.get_output(current_layer, params)
        y_hat = current_layer

        current_layer = X
        for layer, params in zip(self.layers, self.params_shared):
            current_layer = layer.get_output(
                current_layer, params, testing=True)
        y_hat_test = current_layer

        train_loss = self.loss.get_train_loss(y_hat, y, self.params_shared)
        test_loss = self.loss.get_test_loss(y_hat_test, y, self.params_shared)
        test_acc = self.accuracy.get_accuracy(y_hat_test, y)

        preds = y_hat.flatten(self.num_output_dims)

        all_params = [p for p in self.params_shared if p is not None]
        all_params = [p for p_list in all_params for p in p_list]
        updates = list(self.optimizer.get_updates(all_params, train_loss))

        for layer in self.layers:
            if layer.updates is not None:
                for update in layer.updates:
                    updates.append(update)
        updates = tuple(updates)

        self.train_loss_theano = theano.function([X, y], train_loss)
        self.test_loss_theano = theano.function([X, y], test_loss)
        self.test_acc_theano = theano.function([X, y], test_acc)
        self.predict_theano = theano.function([X], preds)
        self.update_step = theano.function(
            inputs=[X, y],
            outputs=train_loss,
            updates=updates)

    def set_final_output_shape(self, output_shape):
        for layer in reversed(self.layers):
            if layer.mods_io_dim:
                layer.set_final_output_shape(output_shape)
                return

    def predict(self, X):
        X = atleast_4d(X)
        return self.predict_theano(X)

    def get_train_loss(self, X, y):
        X, y = atleast_4d(X), atleast_4d(y)
        return self.train_loss_theano(X, y)

    def get_test_loss(self, X, y):
        X, y = atleast_4d(X), atleast_4d(y)
        return self.test_loss_theano(X, y)

    def get_accuracy(self, X, y, batch_size=128, shuffle=True):
        X, y = atleast_4d(X), atleast_4d(y)
        X_batches, y_batches, num_batches = make_batches(
            X, y, batch_size, shuffle=shuffle)
        accuracy = 0
        for X_batch, y_batch in zip(X_batches, y_batches):
            accuracy += self.test_acc_theano(X_batch, y_batch)
        return accuracy / num_batches * 100

    def write_progress(self, epoch, num_epochs, batch_num, num_batches,
                       time, loss):
        progress = ("\rEpoch {}/{} | Batch {}/{} | Time: {}s | Loss: {}   "
                    .format(epoch + 1, num_epochs,
                            batch_num + 1, num_batches,
                            round(time, 1),
                            loss))
        sys.stdout.write(progress)

    def est_time_remaining(self, last_time, iteration, num_iterations):
        iterations_left = num_iterations - iteration - 1
        return last_time * iterations_left

    def compile_dream(self, X_train, dream_state, initializer):
        self.dream_compiled = True
        X_dream_shape = list(X_train.shape)
        X_dream_shape[0] = 1
        X_dream_shape[1] -= len(dream_state)
        X_dream = initializer(tuple(X_dream_shape))
        self.X_dream = theano.shared(atleast_4d(np.append(dream_state, X_dream).astype('float32')))

        current_layer = self.X_dream
        T.set_subtensor(current_layer[:, len(dream_state):, :], Activations.softmax(current_layer[:, len(dream_state):, :]))
        for layer, params in zip(self.layers, self.params_shared):
            current_layer = layer.get_output(
                current_layer, params, testing=True)
        y_hat_dream = current_layer.flatten(1)

        self.optimizer.build([[self.X_dream.get_value()]])

        dream_updates = list(self.optimizer.get_updates([self.X_dream], -y_hat_dream[0]))
        original_var = dream_updates[1][0][:, len(dream_state):, :]
        new_var = dream_updates[1][1][:, len(dream_state):, :]
        dream_updates[1] = (self.X_dream, T.set_subtensor(original_var, new_var))
        self.dream_update = theano.function(
            inputs=[],
            outputs=y_hat_dream,
            updates=dream_updates
        )

    def softmax(self, w, t=1.0):
        e = np.exp(np.array(w) / t)
        dist = e / np.sum(e)
        return dist

    def dream(self, X_train, dream_state, initializer=Initializers.glorot_uniform):
        X_train = atleast_4d(X_train)
        if not self.dream_compiled:
            self.compile_dream(X_train, dream_state, initializer)
        X_dream_shape = list(X_train.shape)
        X_dream_shape[0] = 1
        X_dream_shape[1] -= len(dream_state)
        best_reward = 0
        for i in range(10):
            new_value = self.X_dream.get_value()
            new_value[:, len(dream_state):, :] = atleast_4d(initializer(tuple(X_dream_shape)))
            self.X_dream.set_value(new_value)
            for j in range(100):
                reward = self.dream_update()
            if reward > best_reward:
                best_reward = reward
                best_output = self.X_dream.get_value()
            print('Epoch {}: {}'.format(i, best_reward))
        final_output = best_output[:, len(dream_state):, :]
        final_output = self.softmax(final_output)
        print(final_output)
        print(np.argmax(final_output))
        print(self.predict(np.append(dream_state, final_output)))

    def train(self, X, y, num_epochs=12, shuffle=True,
              params_filename='okapi_params.pk',
              initialize_params=True,
              batch_size=128):
        self.num_output_dims = y.ndim
        X, y = atleast_4d(X), atleast_4d(y)
        if not self.compiled:
            self.compile(X, y, initialize_params=initialize_params)
        X_batches, y_batches, num_batches = make_batches(
            X, y, batch_size, shuffle)
        print('Started training...')
        for epoch in range(0, num_epochs):
            epoch_start = time.clock()
            if shuffle:
                X_batches, y_batches, num_batches = make_batches(
                    X, y, batch_size, shuffle=True)
            total_loss = 0
            for X_batch, y_batch, batch_num in zip(
                    X_batches, y_batches, range(0, num_batches)):
                batch_start = time.clock()
                loss = self.update_step(X_batch, y_batch)
                total_loss += loss
                batch_time = time.clock() - batch_start
                time_rem = self.est_time_remaining(
                    batch_time, batch_num, num_batches)
                self.write_progress(epoch, num_epochs,
                                    batch_num, num_batches,
                                    time_rem, loss)

            epoch_time = time.clock() - epoch_start
            avg_loss = total_loss / num_batches
            self.write_progress(epoch, num_epochs,
                                num_batches - 1, num_batches,
                                epoch_time, avg_loss)
            print()
