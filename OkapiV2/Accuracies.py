import theano.tensor as T


class Accuracy():
    def __init__(self):
        return

    def get_hyperparams_shape(self, max_layer_size, input_shape):
        return None

    def set_hyperparams(self, hyperparams):
        return


class Categorical(Accuracy):
    def get_accuracy(self, y_hat, y):
        return T.mean(T.eq(
            T.argmax(y.flatten(2), axis=1),
            T.argmax(y_hat.flatten(2), axis=1)))
