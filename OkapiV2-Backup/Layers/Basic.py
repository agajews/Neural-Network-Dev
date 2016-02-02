from OkapiV2 import Initializers
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano
import numpy as np


class Layer():
    def __init__(self):
        raise NotImplementedError

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return None

    def set_hyperparams(self, hyperparams):
        return

    def get_init_params(self, input_shape):
        return None

    def get_output_dim(self, input_shape):
        return input_shape


class FullyConnected(Layer):
    def __init__(self, nodes_shape=(1, 1, 1, 1),
                 initializer=Initializers.glorot_uniform,
                 bias_initializer=Initializers.zeros):
        self.nodes_shape = nodes_shape
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        self.updates = None
        self.mods_io_dim = True

    def get_init_params(self, input_shape):
        self.num_nodes = self.nodes_shape[0]
        for i in range(1, len(self.nodes_shape)):
            self.num_nodes *= self.nodes_shape[i]

        num_features = input_shape[1]
        for i in range(2, len(input_shape)):
            num_features *= input_shape[i]

        W_shape = (num_features,) + (self.num_nodes,)
        b_shape = (self.num_nodes,)

        init_params = []
        init_params.append(self.initializer(W_shape))
        init_params.append(self.bias_initializer(b_shape))
        return init_params

    def set_final_output_shape(self, output_shape):
        self.nodes_shape = tuple(output_shape[1:])
        for i in range(4 - len(output_shape) + 1):
            self.nodes_shape += (1, )

    def get_output_dim(self, input_shape):
        self.output_dim = (input_shape[0],) + self.nodes_shape
        return self.output_dim

    def get_output(self, input, params, testing=False):
        num_examples = input.shape[0]
        return (input.flatten(2).dot(params[0]) + params[1]) \
            .reshape((num_examples,) + self.nodes_shape)


class Dropout(Layer):
    def __init__(self, proportion=0.5):
        self.proportion = theano.shared(np.float32(proportion))
        self.rng = RandomStreams(np.random.RandomState(12345).randint(999999))
        self.updates = None
        self.mods_io_dim = False

    def get_output_dim(self, input_shape):
        return input_shape

    def get_output(self, input, params, testing=False):
        if not testing:
            input *= self.rng.binomial(size=input.shape,
                                       p=self.proportion,
                                       dtype='float32')
            return input / self.proportion
        else:
            return input


class BatchNorm(Layer):
    def __init__(self, norm_dim=0, momentum=0.9, epsilon=1e-7,
                 initializer=Initializers.uniform,
                 bias_initializer=Initializers.zeros):
        self.norm_dim = norm_dim
        self.momentum = theano.shared(np.float32(momentum))
        self.epsilon = theano.shared(np.float32(epsilon))
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        self.updates = None
        self.mods_io_dim = False

    def get_init_params(self, input_shape):
        num_features = input_shape[1]
        for i in range(2, len(input_shape)):
            num_features *= input_shape[i]

        self.running_mean = theano.shared(np.zeros((num_features),).astype('float32'))
        self.running_std = theano.shared(np.ones((num_features),).astype('float32'))

        W_shape = (num_features,)
        b_shape = (num_features,)

        init_params = []
        init_params.append(self.initializer(W_shape))
        init_params.append(self.bias_initializer(b_shape))
        return init_params

    def get_output_dim(self, input_shape):
        return input_shape

    def get_output(self, input, params, testing=False):
        W = params[0]
        b = params[1]

        x = input.flatten(2)

        mean = x.mean(self.norm_dim)
        std = x.std(self.norm_dim)

        mean_update = (self.momentum * self.running_mean + (1-self.momentum) * mean).astype('float32')
        std_update = (self.momentum * self.running_std + (1-self.momentum) * std).astype('float32')

        self.updates = [(self.running_mean, mean_update), (self.running_std, std_update)]

        '''self.running_mean = (1-self.momentum) * self.running_mean + \
            self.momentum * mean
        self.running_std = (1-self.momentum) * self.running_std + \
            self.momentum * std'''

        output = (x - self.running_mean) / \
            (self.running_std + self.epsilon) * W + b

        return output.reshape(input.shape).astype('float32')
