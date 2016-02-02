import theano
import theano.tensor as T
import numpy as np


class Optimizer():
    def __init__(self):
        raise NotImplementedError

    def build(self, all_params):
        return


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.99,
                 epsilon=1e-8):
        self.learning_rate = theano.shared(np.float32(learning_rate))
        self.momentum = theano.shared(np.float32(momentum))
        self.epsilon = theano.shared(np.float32(epsilon))

    def build(self, all_params):
        self.accumulators = []
        for params in all_params:
            self.accumulators.append(
                theano.shared(np.zeros(params.get_value().shape)
                              .astype('float32')))

    def get_updates(self, params_list, loss):
        self.updates = []
        for params, accumulator in zip(params_list, self.accumulators):
            grad = T.grad(loss, params)
            accumulator_update = self.momentum * accumulator + \
                (1 - self.momentum) * grad ** 2
            self.updates.append((accumulator, accumulator_update))
            new_params = params - self.learning_rate * grad / \
                T.sqrt(accumulator_update + self.epsilon)
            self.updates.append((params, new_params))
        return self.updates


class SGD(Optimizer):
    def __init__(self, learning_rate=0.001):
        self.learning_rate = theano.shared(np.float32(learning_rate))

    def get_updates(self, params_list, loss):
        self.updates = []
        for params in params_list:
            grad = T.grad(loss, params)
            new_params = params - self.learning_rate * grad
            self.updates.append((params, new_params))
        return self.updates
