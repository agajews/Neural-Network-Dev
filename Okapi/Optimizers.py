import theano
import theano.tensor as T
import numpy as np


class Optimizer():
    def __init__(self):
        raise NotImplementedError


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.99,
                 epsilon=1e-8):
        self.learning_rate = theano.shared(np.float32(learning_rate))
        self.momentum = theano.shared(np.float32(momentum))
        self.epsilon = theano.shared(np.float32(epsilon))
        self.updates = ()

    def setup(self, init_params):
        self.accumulators = []
        for layer_params in init_params:
            if layer_params is not None:
                for params in layer_params:
                    self.accumulators.append(
                        theano.shared(np.zeros(params.get_value().shape)
                                      .astype('float32')))

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return [('double', 0, 0.1),
                ('double', 0.5, 1),
                ('double', 0, 0.01)]

    def set_hyperparams(self, hyperparams):
        self.learning_rate = theano.shared(np.float32(hyperparams[0]))
        self.momentum = theano.shared(np.float32(hyperparams[1]))
        self.epsilon = theano.shared(np.float32(hyperparams[2]))

    def get_updates(self, params_list, grads, loss):
        self.updates = []
        for params, grad, accumulator in zip(params_list, grads,
                                             self.accumulators):
            new_accumulator = self.momentum * accumulator + \
                (1 - self.momentum) * grad ** 2
            self.updates.append((accumulator, new_accumulator))
            new_params = params - self.learning_rate * grad / \
                T.sqrt(new_accumulator + self.epsilon)
            self.updates.append((params, new_params))
        self.updates = tuple(self.updates)
        return self.updates


class SGD(Optimizer):
    def __init__(self, learning_rate=0.001):
        self.learning_rate = theano.shared(np.float32(learning_rate))
        self.updates = ()

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return [('double', 0, 1)]

    def setup(self, init_params):
        return

    def set_hyperparams(self, hyperparams):
        self.learning_rate = theano.shared(np.float32(hyperparams[0]))

    def get_updates(self, params_list, grads, loss):
        self.updates = []
        for params, grad in zip(params_list, grads):
            new_params = params - self.learning_rate * grad
            self.updates.append((params, new_params))
        self.updates = tuple(self.updates)
        return self.updates
