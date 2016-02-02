import theano.tensor as T


def softmax(input):
    return T.nnet.softmax(input.flatten(2))


def alt_softmax(input):
    input_flat = input.flatten(2)
    e_x = T.exp(input_flat - input_flat.max(axis=1, keepdims=True))
    out = e_x / e_x.sum(axis=1, keepdims=True)
    return T.maximum(1e-7, out - 1e-7)


def log_softmax(input):
    diff = input.flatten(2) - T.max(input.flatten(2))
    out = diff - T.log(T.sum(T.exp(diff)))
    return out


def binary(input):
    return T.switch(input > 0, 1e-7, 1)


def softplus(input):
    return T.nnet.softplus(input.flatten(2))


def tanh(input):
    return T.tanh(input)


def hard_sigmoid(input):
    return T.nnet.hard_sigmoid(input)


def sigmoid(input):
    return T.nnet.sigmoid(input)


def ReLU(input):
    return (T.nnet.relu(input) + 1e-7).astype('float32')
