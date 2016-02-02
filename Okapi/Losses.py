import theano
import theano.tensor as T
import numpy as np


class Loss():
    def __init__(self):
        return

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return None

    def set_hyperparams(self, hyperparams):
        return


class CrossentropyTrain(Loss):
    def __init__(self, l1_param=0.00, l2_param=0.00):
        self.l1_param = theano.shared(np.float32(l1_param))
        self.l2_param = theano.shared(np.float32(l2_param))

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return [('double', 0, 10000),
                ('double', 0, 10000)]

    def set_hyperparams(self, hyperparams):
        self.l1_param = hyperparams[0]
        self.l2_param = hyperparams[1]

    def get_loss(self, y_hat, y, params_list):
        num_examples = y_hat.shape[0]
        l1_term = 0
        l2_term = 0
        for layer_params in params_list:
            if layer_params is not None:
                for params in layer_params:
                    l1_term += 1./num_examples * self.l1_param/2 * \
                        T.sum(T.sqr(params))

                    l2_term += 1./num_examples * self.l2_param/2 * \
                        T.sum(T.abs_(params))
        loss = T.nnet.categorical_crossentropy(y_hat.flatten(2),
                                               y.flatten(2)).mean()
        return loss + l1_term + l2_term


class CrossentropyTest(Loss):
    def get_loss(self, y_hat, y, params_list):
        return T.nnet.categorical_crossentropy(
            y_hat.flatten(2), y.flatten(2)).mean()
