from OkapiV2 import Initializers


class RBM(Layer):
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


