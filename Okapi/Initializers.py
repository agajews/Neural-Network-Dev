import numpy as np
import sys
import time


def get_fans(dims):
    fan_in = dims[0] if len(dims) == 2 else np.prod(dims[1:])
    fan_out = dims[1] if len(dims) == 2 else dims[0]
    return fan_in, fan_out


def uniform(dims, std):
    range = (np.sqrt(3) * std, -np.sqrt(3) * std)
    return np.random.uniform(range[0], range[1], dims)


def normal(dims, scale=0.5):
    return np.random.standard_normal(dims) * scale


class Initializer():
    def __init__(self):
        self.is_simple = True

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return None

    def set_hyperparams(self, hyperparams):
        return


class NormalInit(Initializer):
    def get_pre_init_params(self, dims):
        return normal(dims, 1 / np.sqrt(dims[0]))


class GlorotUniformInit(Initializer):
    def __init__(self, gain=1.0):
        self.gain = gain
        self.is_simple = True

    def get_pre_init_params(self, dims):
        fan_in, fan_out = get_fans(dims)
        std = self.gain * np.sqrt(6.0 / (fan_in + fan_out))
        return uniform(dims, std)


class GlorotNormalInit(Initializer):
    def __init__(self, gain=1.0):
        self.gain = gain
        self.is_simple = True

    def get_pre_init_params(self, dims):
        fan_in, fan_out = get_fans(dims)
        std = self.gain * np.sqrt(6.0 / (fan_in + fan_out))
        return normal(dims, std)


class MultiInit(Initializer):
    def __init__(self, num_iterations=100,
                 initializer=GlorotUniformInit()):
        self.is_simple = False
        self.num_iterations = num_iterations
        self.initializer = initializer

    def get_pre_init_params(self, dims):
        return self.initializer.get_pre_init_params(dims)

    def get_init_params(self, dims, X, y, model):
        best_loss = 10000
        best_params = model.init_params
        X_batches, y_batches = model.make_batches(X, y)
        for iteration in range(self.num_iterations):
            start_time = time.clock()
            params = model.get_simple_init_params(model.param_dims)
            model.update_shared_params(params)
            loss = 0
            for X_batch, y_batch in zip(X_batches, y_batches):
                loss += model.get_test_loss(X_batch, y_batch)
            loss /= model.num_batches
            if loss < best_loss:
                best_params = params
                best_loss = loss
            total_time = time.clock() - start_time
            time_remaining = total_time * (self.num_iterations - iteration - 1)
            sys.stdout.write(
                "\rIteration: {}/{} | ETR: {}s | Best Initial Loss: {}    "
                .format(iteration + 1, self.num_iterations,
                        round(time_remaining, 1),
                        best_loss))
        return best_params
