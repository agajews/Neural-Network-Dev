import theano.tensor as T
from theano.tensor.signal import downsample
from theano.sandbox.cuda import dnn
from OkapiV2.Layers.Basic import Layer
from OkapiV2 import Initializers


class Convolutional(Layer):
    def __init__(self, num_filters=1, num_rows=1, num_cols=1,
                 row_stride=1, col_stride=1, pad=False,
                 initializer=Initializers.glorot_uniform):
        self.num_filters = num_filters
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.row_stride = row_stride
        self.col_stride = col_stride
        self.pad = pad
        self.initializer = initializer
        self.updates = None
        self.mods_io_dim = True

    def get_init_params(self, input_shape):
        f_shape = (self.num_filters, input_shape[1],
                   self.num_rows, self.num_cols)
        init_params = []
        init_params.append(self.initializer(f_shape))
        return init_params

    def set_final_output_shape(self, output_shape):
        raise Exception('Convolutional layer cannot be last layer that \
                         modifies the dimension of its input')

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
        if dnn.dnn_available():
            return dnn.dnn_conv(img=input,
                                kerns=params[0],
                                subsample=(self.row_stride, self.col_stride),
                                border_mode=self.conv_mode)
        else:
            return T.nnet.conv2d(input,
                                 params[0],
                                 subsample=(self.row_stride, self.col_stride),
                                 border_mode=self.conv_mode)


class MaxPooling(Layer):
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
        self.updates = None
        self.mods_io_dim = True

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
