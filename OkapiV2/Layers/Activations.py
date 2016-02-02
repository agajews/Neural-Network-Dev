import theano.tensor as T
from OkapiV2 import Activations, Initializers
from OkapiV2.Layers.Basic import Layer


class ActivationLayer(Layer):
    def __init__(self, activation=Activations.tanh):
        self.activation = activation
        self.updates = None
        self.mods_io_dim = False

    def get_output_dim(self, input_shape):
        return input_shape

    def get_output(self, input, params, testing=False):
        return self.activation(input)


class PReLULayer(Layer):
    def __init__(self, initializer=Initializers.zeros):
        self.initializer = initializer
        self.updates = None
        self.mods_io_dim = False

    def get_init_params(self, input_shape):
        W_shape = input_shape[1:]
        init_params = []
        init_params.append(self.initializer(W_shape))
        return init_params

    def get_output_dim(self, input_shape):
        self.output_shape = input_shape
        return self.output_shape

    def get_output(self, input, params, testing=False):
        output = (0.5 * (1 + params[0])) * input + \
                 (0.5 * (1 + params[0])) * T.abs_(input)
        return (output + 1e-7).astype('float32')


'''class SoftmaxLayer(Layer):
    def __init__(self):
        self.updates = None
        self.mods_io_dim = False

    def get_output(self, input, params, testing=False):
        if testing:
            return Activations.alt_softmax(input)
        else:
            input = input.flatten(2)
            xdev = input-input.max(1, keepdims=True)
            lsm = xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))
            return lsm'''
