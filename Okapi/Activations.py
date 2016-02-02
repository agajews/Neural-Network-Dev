import theano.tensor as T


def softmax(input):
    return T.nnet.softmax(input.flatten(2))


def tanh(input):
    return T.tanh(input)


def hard_sigmoid(input):
    return T.nnet.hard_sigmoid(input)


def sigmoid(input):
    return T.nnet.sigmoid(input)


def ReLU(input):
    return (0.5 * (input + T.abs_(input)))
