import theano
import theano.tensor as T
from OkapiV2 import Activations, Initializers
from OkapiV2.Layers.Basic import Layer


class RecurrentLayer(Layer):
    def __init__(self):
        raise NotImplementedError

    def get_init_params(self, input_shape):
        self.num_nodes = self.nodes_shape[0]
        for i in range(1, len(self.nodes_shape)):
            self.num_nodes *= self.nodes_shape[i]

        self.num_features = input_shape[2]
        for i in range(3, len(input_shape)):
            self.num_features *= input_shape[i]

        self.init_params = self.get_init_params_list(self.num_features,
                                                     self.num_nodes)
        return self.init_params

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


class SimpleRecurrent(RecurrentLayer):
    def __init__(self, nodes_shape=(1, 1, 1, 1), activation=Activations.tanh,
                 return_sequences=False,
                 initializer=Initializers.glorot_uniform,
                 inner_initializer=Initializers.orthogonal,
                 bias_initializer=Initializers.zeros):
        self.nodes_shape = nodes_shape
        self.activation = activation
        self.return_sequences = return_sequences
        self.initializer = initializer
        self.inner_initializer = inner_initializer
        self.bias_initializer = bias_initializer
        self.updates = None
        self.mods_io_dim = True

    def get_init_params_list(self, num_features, num_nodes):
        W_shape = (num_features,) + (num_nodes,)
        U_shape = (num_nodes,) + (num_nodes,)
        b_shape = (num_nodes,)

        init_params = []
        init_params.append(self.initializer(W_shape))
        init_params.append(self.inner_initializer(U_shape))
        init_params.append(self.bias_initializer(b_shape))
        return init_params

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


class GRU(RecurrentLayer):
    def __init__(self, nodes_shape=(1, 1, 1, 1), activation=Activations.tanh,
                 return_sequences=False,
                 initializer=Initializers.glorot_uniform,
                 inner_initializer=Initializers.orthogonal,
                 bias_initializer=Initializers.zeros):
        self.nodes_shape = nodes_shape
        self.activation = activation
        self.return_sequences = return_sequences
        self.initializer = initializer
        self.inner_initializer = inner_initializer
        self.bias_initializer = bias_initializer
        self.updates = None
        self.mods_io_dim = True

    def get_init_params_list(self, num_features, num_nodes):
        W_z_shape = (num_features,) + (num_nodes,)
        U_z_shape = (num_nodes,) + (num_nodes,)
        b_z_shape = (num_nodes,)

        W_r_shape = (num_features,) + (num_nodes,)
        U_r_shape = (num_nodes,) + (num_nodes,)
        b_r_shape = (num_nodes,)

        W_c_shape = (num_features,) + (num_nodes,)
        U_c_shape = (num_nodes,) + (num_nodes,)
        b_c_shape = (num_nodes,)

        init_params = []
        init_params.append(self.initializer(W_z_shape))
        init_params.append(self.inner_initializer(U_z_shape))
        init_params.append(self.bias_initializer(b_z_shape))

        init_params.append(self.initializer(W_r_shape))
        init_params.append(self.inner_initializer(U_r_shape))
        init_params.append(self.bias_initializer(b_r_shape))

        init_params.append(self.initializer(W_c_shape))
        init_params.append(self.inner_initializer(U_c_shape))
        init_params.append(self.bias_initializer(b_c_shape))
        return init_params

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


class LSTM(RecurrentLayer):
    def __init__(self, nodes_shape=(1, 1, 1, 1),
                 inner_activation=Activations.hard_sigmoid,
                 activation=Activations.tanh,
                 return_sequences=False,
                 initializer=Initializers.glorot_uniform,
                 inner_initializer=Initializers.orthogonal,
                 forget_initializer=Initializers.ones,
                 bias_initializer=Initializers.zeros):
        self.nodes_shape = nodes_shape
        self.inner_activation = inner_activation
        self.activation = activation
        self.return_sequences = return_sequences
        self.initializer = initializer
        self.inner_initializer = inner_initializer
        self.forget_initializer = forget_initializer
        self.bias_initializer = bias_initializer
        self.updates = None
        self.mods_io_dim = True

    def get_init_params_list(self, num_features, num_nodes):
        W_i_shape = (num_features,) + (num_nodes,)
        U_i_shape = (num_nodes,) + (num_nodes,)
        b_i_shape = (num_nodes,)

        W_f_shape = (num_features,) + (num_nodes,)
        U_f_shape = (num_nodes,) + (num_nodes,)
        b_f_shape = (num_nodes,)

        W_c_shape = (num_features,) + (num_nodes,)
        U_c_shape = (num_nodes,) + (num_nodes,)
        b_c_shape = (num_nodes,)

        W_o_shape = (num_features,) + (num_nodes,)
        U_o_shape = (num_nodes,) + (num_nodes,)
        b_o_shape = (num_nodes,)

        init_params = []
        init_params.append(self.initializer(W_i_shape))
        init_params.append(self.inner_initializer(U_i_shape))
        init_params.append(self.bias_initializer(b_i_shape))

        init_params.append(self.initializer(W_f_shape))
        init_params.append(self.inner_initializer(U_f_shape))
        init_params.append(self.forget_initializer(b_f_shape))

        init_params.append(self.initializer(W_c_shape))
        init_params.append(self.inner_initializer(U_c_shape))
        init_params.append(self.bias_initializer(b_c_shape))

        init_params.append(self.initializer(W_o_shape))
        init_params.append(self.inner_initializer(U_o_shape))
        init_params.append(self.bias_initializer(b_o_shape))
        return init_params

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
