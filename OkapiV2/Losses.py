import theano
import theano.tensor as T
import numpy as np


class Regularizer():
    def __init__(self):
        return


class L1Reg(Regularizer):
    def __init__(self, param=0.0):
        self.param = param

    def get_reg_term(self, params_list, num_examples):
        l1_term = 0
        for params in params_list:
            l1_term += 1. / num_examples * self.l1_param / 2 * \
                T.sum(T.sqr(params))
        return l1_term

class L2Reg(Regularizer):
    def __init__(self, param=0.0):
        self.param = param

    def get_reg_term(self, params_list, num_examples):
        l2_term = 0
        for params in params_list:
            l2_term += 1. / num_examples * self.l2_param / 2 * \
                T.sum(T.abs_(params))
        return l2_term

class Loss():
    def __init__(self, regularizer=None):
        self.regularizer = regularizer


class Crossentropy(Loss):
    def get_train_loss(self, y_hat, y, params_list):
        num_examples = y_hat.shape[0]
        reg_term = 0
        if self.regularizer is not None:
            reg_term = self.regularizer.get_reg_term(params_list, num_examples)
        # y_hat /= y_hat.sum(axis=-1, keepdims=True)
        y_hat = T.clip(y_hat, 1e-7, 1.0 - 1e-7)
        loss = T.nnet.categorical_crossentropy(
            y_hat.flatten(2), y.flatten(2)).mean()  # Maybe no y_hat.flatten?
        return loss + reg_term

    def get_test_loss(self, y_hat, y, params):
        return T.nnet.categorical_crossentropy(
            y_hat.flatten(2), y.flatten(2)).mean()


class MeanSquared(Loss):
    def get_train_loss(self, y_hat, y, params_list):
        num_examples = y_hat.shape[0]
        reg_term = 0
        if self.regularizer is not None:
            reg_term = self.regularizer.get_reg_term(params_list, num_examples)
        return T.mean(T.pow((y.flatten(2) - y_hat.flatten(2)), 2)) + reg_term

    def get_test_loss(self, y_hat, y, params_list):
        return T.mean(T.pow((y.flatten(2) - y_hat.flatten(2)), 2))


'''class SoftmaxCrossentropy(Loss):
    def __init__(self, l1_param=0.00, l2_param=0.00):
        self.l1_param = theano.shared(np.float32(l1_param))
        self.l2_param = theano.shared(np.float32(l2_param))

    def get_train_loss(self, y_hat, y, params_list):
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
        loss = T.nnet.categorical_crossentropy(
            y_hat.flatten(2), y.flatten(2)).mean()
        y, y_hat = y.flatten(2), y_hat.flatten(2)
        y_hat = y_hat - y_hat.max(axis=1).dimshuffle(0, 'x')
        log_prob = y_hat - T.log(T.exp(y_hat).sum(axis=1).dimshuffle(0, 'x'))
        loss = -T.mean((log_prob * y).sum(axis=1))
        loss = T.nnet.categorical_crossentropy(
            T.nnet.softmax(y_hat.flatten(2)),
            y.flatten(2)).mean()
        y = y.flatten(2)
        xdev = y_hat-y_hat.max(1, keepdims=True)
        y_hat = xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))
        loss = -T.sum(y*y_hat, axis=1).mean()
        loss = T.nnet.categorical_crossentropy(
            y_hat.flatten(2), y.flatten(2)).mean()
        y_hat /= y_hat.sum(axis=-1, keepdims=True)
        y_hat = T.clip(y_hat, 10e-8, 1.0 - 10e-8)
        loss = T.mean(T.nnet.categorical_crossentropy(y_hat, y), axis=-1)
        xdev = y_hat-y_hat.max(1, keepdims=True)
        lsm = xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))
        sm = T.exp(lsm)
        # loss = T.nnet.categorical_crossentropy(y_hat, y).mean()
        diff = y_hat.flatten(2) - T.max(y_hat.flatten(2))
        y_hat = diff - T.log(T.sum(T.exp(diff)))
        loss = -T.sum(y * y_hat, axis=1).mean()
        loss = -T.sum(y * y_hat, axis=1).mean()
        return loss + l1_term + l2_term

    def get_test_loss(self, y_hat, y, params_list):
        return T.nnet.categorical_crossentropy(
            y_hat.flatten(2), y.flatten(2)).mean()'''


'''class AltSoftmaxLoss(Loss):
    def get_train_loss(self, y_hat, y, params_list):
        y_flat = y.flatten(2)
        y_hat_flat = y_hat.flatten(2)
        xdev = y_hat_flat - y_hat_flat.max(1, keepdims=True)
        lsm = xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))
        return -T.sum(y_flat * lsm, axis=1).mean()

    def get_test_loss(self, y_hat, y, params_list):
        return T.nnet.categorical_crossentropy(
            y_hat.flatten(2), y.flatten(2)).mean()'''
