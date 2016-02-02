import numpy as np
import sklearn
import sklearn.datasets
import theano
import theano.tensor as T

theano.config.floatX = 'float32'
theano.config.allow_input_downcast = True

X_train, y_train = sklearn.datasets.make_moons(50000, noise=0.20)
y_train = np.eye(2)[y_train]

input_dim = X_train.shape[1]
num_examples = X_train.shape[0]
output_dim = 2

HIDDEN_DIM = 100
ALPHA = np.float32(0.02)
LAMBDA = np.float32(0.01)
BATCH_SIZE = 2

X = T.matrix('X')
y = T.matrix('y')

W1 = theano.shared(np.random.randn(input_dim, HIDDEN_DIM)
                   .astype('float32'), name='W1')
b1 = theano.shared(np.zeros(HIDDEN_DIM)
                   .astype('float32'), name='b1')

W2 = theano.shared(np.random.randn(HIDDEN_DIM, HIDDEN_DIM)
                   .astype('float32'), name='W2')
b2 = theano.shared(np.zeros(HIDDEN_DIM)
                   .astype('float32'), name='b2')

W3 = theano.shared(np.random.randn(HIDDEN_DIM, output_dim)
                   .astype('float32'), name='W3')
b3 = theano.shared(np.zeros(output_dim)
                   .astype('float32'), name='b3')


z1 = X.dot(W1) + b1
a1 = T.tanh(z1)
z2 = a1.dot(W2) + b2
a2 = T.tanh(z2)
z3 = a2.dot(W3) + b3
hyp = T.nnet.softmax(z3)

loss_reg = 1./num_examples * LAMBDA/2 *\
    (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2)) + T.sum(T.sqr(W3)))
loss = T.nnet.categorical_crossentropy(hyp, y).mean() + loss_reg

dW2 = T.grad(loss, W2)
db2 = T.grad(loss, b2)
dW1 = T.grad(loss, W1)
db1 = T.grad(loss, b1)

calculate_loss = theano.function([X, y], loss, allow_input_downcast=True)
gradient_step = theano.function(
    inputs=[X, y],
    updates=((W2, W2 - ALPHA * dW2),
             (W1, W1 - ALPHA * dW1),
             (b2, b2 - ALPHA * db2),
             (b1, b1 - ALPHA * db1)),
    allow_input_downcast=True)


def def_batches(X_train=X_train, y_train=y_train, BATCH_SIZE=BATCH_SIZE):
    order = np.random.permutation(y_train.shape[0])
    X_train = X_train[order, :]
    y_train = y_train[order]
    X_batches = np.array_split(X_train, BATCH_SIZE)
    y_batches = np.array_split(y_train, BATCH_SIZE)
    return X_batches, y_batches


def set_random_params(num_initializations, X_batches, y_batches,
                      X_train, y_train, print_loss=True):
    np.random.seed(0)
    min_loss = 10000000
    for i in range(0, int(num_initializations+1)):
        b1.set_value((np.random.randn(HIDDEN_DIM) /
                     np.sqrt(HIDDEN_DIM)).astype('float32'))
        b2.set_value((np.random.randn(HIDDEN_DIM) /
                     np.sqrt(HIDDEN_DIM)).astype('float32'))
        b3.set_value((np.random.randn(output_dim) /
                     np.sqrt(output_dim)).astype('float32'))
        W1.set_value((np.random.randn(input_dim, HIDDEN_DIM) /
                     np.sqrt(input_dim)).astype('float32'))
        W2.set_value((np.random.randn(HIDDEN_DIM, HIDDEN_DIM) /
                     np.sqrt(HIDDEN_DIM)).astype('float32'))
        W3.set_value((np.random.randn(HIDDEN_DIM, output_dim) /
                     np.sqrt(HIDDEN_DIM)).astype('float32'))
        batch = np.random.randint(0, len(y_batches))
        loss = calculate_loss(X_batches[batch].astype('float32'),
                              y_batches[batch].astype('float32'))
        if loss < min_loss:
            b1_init = b1.get_value()
            b2_init = b2.get_value()
            b3_init = b3.get_value()
            W1_init = W1.get_value()
            W2_init = W2.get_value()
            W3_init = W3.get_value()
            min_loss = loss
        if (print_loss and i % 1000 == 0):
            print("Min loss after random initialization {}: {}"
                  .format(i, min_loss))
    b1.set_value(b1_init)
    b2.set_value(b2_init)
    b3.set_value(b3_init)
    W1.set_value(W1_init)
    W2.set_value(W2_init)
    W3.set_value(W3_init)


def train_network(num_epochs, X_train=X_train, y_train=y_train,
                  print_loss=True):
    np.random.seed(1)
    X_batches, y_batches = def_batches()
    set_random_params(num_epochs, X_batches, y_batches, X_train, y_train)

    for i in range(0, num_epochs+1):
        for batch in range(0, len(y_batches)):
            gradient_step(X_batches[batch].astype('float32'),
                          y_batches[batch].astype('float32'))
        if print_loss and i % 1000 == 0:
            print("Loss after epoch {}: {}".format(i,
                  calculate_loss(X_train.astype('float32'),
                                 y_train.astype('float32'))))

train_network(20000)
