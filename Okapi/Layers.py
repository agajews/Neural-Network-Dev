import theano.tensor as T
import theano
from theano.tensor.signal import downsample
import Okapi.Activations as Activations
import numpy as np


class Layer():
    def __init__(self):
        raise NotImplementedError

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return None

    def set_hyperparams(self, hyperparams):
        return

    def get_param_dims(self, input_shape):
        return None

    def get_output_dim(self, input_shape):
        return input_shape


class FullyConnectedLayer(Layer):
    def __init__(self, nodes_shape=(1, 1, 1, 1)):
        self.nodes_shape = nodes_shape
        self.takes_params = True
        self.mods_io_dim = True
        self.enabled_for_testing = True

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return [('int', 1, max_layer_size),
                ('int', 1, max_layer_size),
                ('int', 1, max_layer_size),
                ('int', 1, max_layer_size)]

    def set_hyperparams(self, hyperparams):
        self.nodes_shape = [1, 1, 1, 1]
        for i in range(0, 3):
            self.nodes_shape[i] = hyperparams[i]
        self.nodes_shape = tuple(self.nodes_shape)

    def get_param_dims(self, input_shape):
        self.num_nodes = self.nodes_shape[0]
        for i in range(1, len(self.nodes_shape)):
            self.num_nodes *= self.nodes_shape[i]

        num_features = input_shape[1]
        for i in range(2, len(input_shape)):
            num_features *= input_shape[i]

        self.param_dims = [(num_features,) + (self.num_nodes,),  # W
                           (self.num_nodes,)]  # b
        return self.param_dims

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


class ConvLayer(Layer):
    def __init__(self, num_filters=1, num_rows=1, num_cols=1,
                 row_stride=1, col_stride=1, pad=False):
        self.num_filters = num_filters
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.row_stride = row_stride
        self.col_stride = col_stride
        self.pad = pad
        self.takes_params = True
        self.mods_io_dim = True
        self.enabled_for_testing = True

    def get_param_dims(self, input_shape):
        self.param_dims = [(self.num_filters, input_shape[1],
                            self.num_rows, self.num_cols)]
        return self.param_dims

    def set_final_output_shape(self, output_shape):
        return

    def get_output_dim(self, input_shape):
        def get_conv_output_length(input_length, filter_size, stride,
                                   pad):
            if not pad:
                output_length = input_length - filter_size + 1
            else:
                output_length = input_length + filter_size - 1
            return (output_length + stride - 1) // stride

        output_rows = get_conv_output_length(input_shape[2],
                                             self.num_rows,
                                             self.row_stride,
                                             self.pad)
        output_cols = get_conv_output_length(input_shape[3],
                                             self.num_cols,
                                             self.col_stride,
                                             self.pad)
        if self.pad:
            self.conv_mode = 'full'
        else:
            self.conv_mode = 'valid'
        self.output_dim = (input_shape[0], self.num_filters,
                           output_rows, output_cols)
        return self.output_dim

    def get_output(self, input, params, testing=False):
        return T.nnet.conv.conv2d(input, params[0],
                                  subsample=(self.row_stride,
                                             self.col_stride),
                                  filter_shape=self.param_dims[0],
                                  border_mode=self.conv_mode)


class MaxPoolingLayer(Layer):
    def __init__(self, pool_rows=1, pool_cols=1, row_stride=1, col_stride=1,
                 pad=(0, 0)):
        self.pool_rows = pool_rows
        self.pool_cols = pool_cols
        self.row_stride = row_stride
        self.col_stride = col_stride
        if pad[0] is 0 and pad[1] is 0:
            self.ignore_border = True
        else:
            self.ignore_border = False
        self.pad = pad
        self.takes_params = True
        self.mods_io_dim = True
        self.enabled_for_testing = True

    def set_final_output_shape(self, output_shape):
        return

    def get_output_dim(self, input_shape):
        def get_pool_output_length(input_length, pool_size, stride,
                                   pad, ignore_border):
            if ignore_border:
                output_length = input_length + 2 * pad - pool_size + 1
                output_length = (output_length + stride - 1) // stride
            else:
                if stride >= pool_size:
                    output_length = (input_length + stride - 1) // stride
                else:
                    output_length = max(
                        0, (input_length - pool_size + stride - 1)
                        // stride) + 1
            return output_length

        output_rows = get_pool_output_length(input_shape[2],
                                             self.pool_rows,
                                             self.row_stride,
                                             self.pad[0],
                                             self.ignore_border)
        output_cols = get_pool_output_length(input_shape[3],
                                             self.pool_cols,
                                             self.col_stride,
                                             self.pad[1],
                                             self.ignore_border)
        self.output_dim = input_shape[0:2] + (output_rows, output_cols)
        return self.output_dim

    def get_output(self, input, params, testing=False):
        return downsample.max_pool_2d(input,
                                      ds=(self.pool_rows, self.pool_cols),
                                      st=(self.row_stride, self.col_stride),
                                      ignore_border=self.ignore_border,
                                      mode='max',
                                      padding=self.pad)


class RecurrentLayer(Layer):
    def __init__(self):
        raise NotImplementedError

    def get_param_dims(self, input_shape):
        self.num_nodes = self.nodes_shape[0]
        for i in range(1, len(self.nodes_shape)):
            self.num_nodes *= self.nodes_shape[i]

        self.num_features = input_shape[2]
        for i in range(3, len(input_shape)):
            self.num_features *= input_shape[i]

        self.param_dims = self.get_param_dims_list(self.num_features,
                                                   self.num_nodes)
        return self.param_dims

    def get_param_dims_list(self, num_features, num_nodes):
        raise NotImplementedError

    def set_final_output_shape(self, output_shape):
        if (output_shape[2] == 1) and (output_shape[3] == 1):
            self.return_sequences = False
        self.nodes_shape = tuple(output_shape[1:])

    def get_output_dim(self, input_shape):
        if not self.return_sequences:
            self.output_dim = (input_shape[0],) + self.nodes_shape
        else:
            self.output_dim = (input_shape[0],) + (input_shape[1],) + \
                self.nodes_shape
        return self.output_dim

    def get_output(self, input, params, testing=False):
        input = input.flatten(3)

        num_examples = input.shape[0]
        sequence_length = input.shape[1]

        outputs, updates = theano.scan(
            self.forward_step,
            sequences=T.arange(sequence_length),
            outputs_info=self.get_init_states(num_examples, self.num_nodes),
            non_sequences=[input] + params
        )

        output = outputs[0]
        if not self.return_sequences:
            return output[-1] \
                .reshape((num_examples,) + self.nodes_shape)
        else:
            return output.dimshuffle((0, 2, 1)) \
                .reshape((num_examples,) + (sequence_length,) +
                         self.nodes_shape)


class SimpleRecurrentLayer(RecurrentLayer):
    def __init__(self, nodes_shape=(1, 1, 1, 1), activation=Activations.tanh,
                 return_sequences=False):
        self.nodes_shape = nodes_shape
        self.takes_params = True
        self.mods_io_dim = True
        self.enabled_for_testing = True
        self.activation = activation
        self.return_sequences = return_sequences

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return [('int', 1, max_layer_size),
                ('int', 1, max_layer_size),
                ('int', 1, max_layer_size),
                ('int', 1, max_layer_size),
                ('activation',),
                ('boolean',)]

    def set_hyperparams(self, hyperparams):
        self.nodes_shape = [1, 1, 1, 1]
        for i in range(0, 3):
            self.nodes_shape[i] = hyperparams[i]
        self.nodes_shape = tuple(self.nodes_shape)
        self.activation = hyperparams[4]
        self.return_sequences = hyperparams[5]

    def get_param_dims_list(self, num_features, num_nodes):
        self.param_dims = [(num_features,) + (num_nodes,),  # W
                           (num_nodes,) + (num_nodes,),  # U
                           (num_nodes,),  # b
                           ]
        return self.param_dims

    def get_init_states(self, num_examples, num_nodes):
        init_states = [None,
                       T.zeros((num_examples, num_nodes))
                       ]
        return init_states

    def forward_step(self, t, s_t_prev, x, W, U, b):
        x_t = x[:, t, :].flatten(2)
        s_t = self.activation((x_t.dot(W) + b) * s_t_prev.dot(U))

        output = s_t
        s_t_prev = s_t
        return output, s_t_prev


class GRULayer(RecurrentLayer):
    def __init__(self, nodes_shape=(1, 1, 1, 1), activation=Activations.tanh,
                 return_sequences=False):
        self.nodes_shape = nodes_shape
        self.takes_params = True
        self.mods_io_dim = True
        self.enabled_for_testing = True
        self.activation = activation
        self.return_sequences = return_sequences

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return [('int', 1, max_layer_size),
                ('int', 1, max_layer_size),
                ('int', 1, max_layer_size),
                ('int', 1, max_layer_size),
                ('activation',),
                ('boolean',)]

    def set_hyperparams(self, hyperparams):
        self.nodes_shape = [1, 1, 1, 1]
        for i in range(0, 3):
            self.nodes_shape[i] = hyperparams[i]
        self.nodes_shape = tuple(self.nodes_shape)
        self.activation = hyperparams[4]
        self.return_sequences = hyperparams[5]

    def get_param_dims_list(self, num_features, num_nodes):
        self.param_dims = [(num_features,) + (num_nodes,),  # W_z
                           (num_nodes, num_nodes),  # U_z
                           (num_nodes,),  # b_z
                           (num_features,) + (num_nodes,),  # W_r
                           (num_nodes, num_nodes),  # U_r
                           (num_nodes,),  # b_r
                           (num_features,) + (num_nodes,),  # W_c
                           (num_nodes, num_nodes),  # U_c
                           (num_nodes,),  # b_c
                           ]
        return self.param_dims

    def get_init_states(self, num_examples, num_nodes):
        init_states = [None,
                       T.zeros((num_examples, num_nodes))
                       ]
        return init_states

    def forward_step(self, t, s_t_prev, x,
                     W_z, U_z, b_z,
                     W_r, U_r, b_r,
                     W_c, U_c, b_c):
        x_t = x.flatten(3)[:, t, :]

        x_z = x_t.dot(W_z) + b_z
        x_r = x_t.dot(W_r) + b_r
        x_c = x_t.dot(W_c) + b_c

        z = self.activation(x_z + s_t_prev.dot(U_z))
        r = self.activation(x_r + s_t_prev.dot(U_r))
        c = self.activation(x_c + (r * s_t_prev).dot(U_c))

        s_t = z * s_t_prev + (1 - z) * c
        output = s_t
        s_t_prev = s_t
        return output, s_t_prev


class LSTMLayer(RecurrentLayer):
    def __init__(self, nodes_shape=(1, 1, 1, 1),
                 inner_activation=Activations.hard_sigmoid,
                 activation=Activations.tanh,
                 return_sequences=False):
        self.nodes_shape = nodes_shape
        self.takes_params = True
        self.mods_io_dim = True
        self.enabled_for_testing = True
        self.inner_activation = inner_activation
        self.activation = activation
        self.return_sequences = return_sequences

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return [('int', 1, max_layer_size),
                ('int', 1, max_layer_size),
                ('int', 1, max_layer_size),
                ('int', 1, max_layer_size),
                ('activation',),
                ('activation',),
                ('boolean',)]

    def set_hyperparams(self, hyperparams):
        self.nodes_shape = [1, 1, 1, 1]
        for i in range(0, 3):
            self.nodes_shape[i] = hyperparams[i]
        self.nodes_shape = tuple(self.nodes_shape)
        self.activation = hyperparams[4]
        self.inner_activation = hyperparams[5]
        self.return_sequences = hyperparams[6]

    def get_param_dims_list(self, num_features, num_nodes):
        self.param_dims = [(num_features,) + (num_nodes,),  # W_i
                           (num_nodes, num_nodes),  # U_i
                           (num_nodes,),  # b_i
                           (num_features,) + (num_nodes,),  # W_f
                           (num_nodes, num_nodes),  # U_f
                           (num_nodes,),  # b_f
                           (num_features,) + (num_nodes,),  # W_c
                           (num_nodes, num_nodes),  # U_c
                           (num_nodes,),  # b_c
                           (num_features,) + (num_nodes,),  # W_o
                           (num_nodes, num_nodes),  # U_o
                           (num_nodes,),  # b_o
                           ]
        return self.param_dims

    def get_init_states(self, num_examples, num_nodes):
        init_states = [None,
                       T.zeros((num_examples, num_nodes)),
                       T.zeros((num_examples, num_nodes)),
                       ]
        return init_states

    def forward_step(self, t, s_t_prev, c_t_prev, x,
                     W_i, U_i, b_i,
                     W_f, U_f, b_f,
                     W_c, U_c, b_c,
                     W_o, U_o, b_o):
        x_t = x.flatten(3)[:, t, :]

        x_i = x_t.dot(W_i) + b_i
        x_f = x_t.dot(W_f) + b_f
        x_c = x_t.dot(W_c) + b_c
        x_o = x_t.dot(W_o) + b_o

        i = self.inner_activation(x_i + s_t_prev.dot(U_i))
        f = self.inner_activation(x_f + s_t_prev.dot(U_f))
        c = f * c_t_prev + i * \
            self.inner_activation(x_c + s_t_prev.dot(U_c))
        o = self.inner_activation(x_o + s_t_prev.dot(U_o))

        h = o * self.activation(c)

        output = h
        s_t_prev = h
        c_t_prev = c
        return output, s_t_prev, c_t_prev


class ActivationLayer(Layer):
    def __init__(self, activation=Activations.tanh):
        self.takes_params = False
        self.mods_io_dim = False
        self.enabled_for_testing = True
        self.activation = activation

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return [('activation',)]

    def set_hyperparams(self, hyperparams):
        self.activation = hyperparams[0]

    def get_output_dim(self, input_shape):
        return input_shape

    def get_output(self, input, params, testing=False):
        return self.activation(input)


class PReLULayer(Layer):
    def __init__(self):
        self.takes_params = True
        self.mods_io_dim = False
        self.enabled_for_testing = True

    def get_param_dims(self, input_shape):
        return [input_shape[1:]]

    def get_output_dim(self, input_shape):
        self.output_shape = input_shape
        return self.output_shape

    def get_output(self, input, params, testing=False):
        output = (0.5 * (1 + params[0])) * input + \
                 (0.5 * (1 + params[0])) * T.abs_(input)
        return output


class DropoutLayer(Layer):
    def __init__(self, proportion=0.5):
        self.takes_params = False
        self.mods_io_dim = False
        self.enabled_for_testing = False
        self.proportion = theano.shared(np.float32(proportion))
        self.rand_streams = T.shared_randomstreams \
            .RandomStreams(np.random.RandomState(12345).randint(999999))

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return [('double', 0, 1)]

    def set_hyperparams(self, hyperparams):
        self.proportion = theano.shared(np.float32(hyperparams[0]))

    def get_output_dim(self, input_shape):
        return input_shape

    def get_output(self, input, params, testing=False):
        if not testing:
            return input * self.rand_streams.binomial(size=input.shape,
                                                      p=self.proportion,
                                                      dtype='float32')
        else:
            return input


class BatchNormalizationLayer(Layer):
    def __init__(self, norm_dim=0, momentum=0.9, epsilon=1e-8):
        self.takes_params = True
        self.mods_io_dim = False
        self.enabled_for_testing = True
        self.norm_dim = norm_dim
        self.momentum = theano.shared(np.float32(momentum))
        self.epsilon = theano.shared(np.float32(epsilon))

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return [('int', 0, 1),
                ('double', 0.5, 1),
                ('double', 0, 0.01)]

    def set_hyperparams(self, hyperparams):
        self.norm_dim = hyperparams[0]
        self.momentum = theano.shared(np.float32(hyperparams[1]))
        self.epsilon = theano.shared(np.float32(hyperparams[2]))

    def get_param_dims(self, input_shape):
        num_features = input_shape[1]
        for i in range(2, len(input_shape)):
            num_features *= input_shape[i]
        self.param_dims = [(num_features,),  # W
                           (num_features,)]  # b
        self.running_mean = T.zeros(num_features)
        self.running_std = T.ones(num_features)
        return self.param_dims

    def get_output_dim(self, input_shape):
        return input_shape

    def get_output(self, input, params, testing=False):
        W = params[0]
        b = params[1]

        x = input.flatten(2)

        mean = x.mean(self.norm_dim, keepdims=True)
        std = x.std(self.norm_dim, keepdims=True)

        self.running_mean = (1-self.momentum) * self.running_mean + \
            self.momentum * mean
        self.running_std = (1-self.momentum) * self.running_std + \
            self.momentum * std

        output = (x - self.running_mean) / \
            (self.running_std + self.epsilon) * W + b

        return output.reshape(input.shape)
