import theano
import theano.tensor as T
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

X_train, y_train = sklearn.datasets.make_moons(20000, noise=0.20)

HIDDEN_DIM = 30
INPUT_DIM = 2
OUTPUT_DIM = 2
ALPHA = np.float32(0.01)
LAMBDA = np.float32(0.01)
BATCH_SIZE = 20
NUM_LAYERS = 3

X = T.matrix('X')
y = T.lvector('y')
num_examples = y.shape[0]

# Train
theta1 = theano.shared(np.random.randn(INPUT_DIM, HIDDEN_DIM)
                       .astype('float32'), name='theta1')
bias1 = theano.shared(np.random.randn(HIDDEN_DIM)
                      .astype('float32'), name='bias1')
thetaH = theano.shared(np.random.randn(HIDDEN_DIM, HIDDEN_DIM, NUM_LAYERS-2)
                       .astype('float32'), name='thetaH')
biasH = theano.shared(np.random.randn(HIDDEN_DIM, NUM_LAYERS-2)
                      .astype('float32'), name='biasH')
thetaL = theano.shared(np.random.randn(HIDDEN_DIM, OUTPUT_DIM)
                       .astype('float32'), name='thetaL')
biasL = theano.shared(np.random.randn(OUTPUT_DIM)
                      .astype('float32'), name='biasL')

layer0 = X
sum1 = layer0.dot(theta1) + bias1
layer1 = T.tanh(sum1)
layerH = layer1
for i in range(0, thetaH.get_value().shape[2]):
    layerH = T.tanh(layerH.dot(thetaH[:, :, i] + biasH[:, i]))
sumL = layerH.dot(thetaL) + biasL
layerL = T.nnet.softmax(sumL)
hyp = layerL

reg_term = 1./num_examples * LAMBDA/2 * (T.sum(T.sqr(theta1)) +
                                         T.sum(T.sqr(thetaH)) +
                                         T.sum(T.sqr(thetaL)))

cost = T.nnet.categorical_crossentropy(hyp, y).mean() + reg_term

prediction = T.argmax(hyp, axis=1)

forward_prop = theano.function([X], hyp)
calculate_cost = theano.function([X, y], cost)
predict = theano.function([X], prediction)

thetaL_grad = T.grad(cost, thetaL)
biasL_grad = T.grad(cost, biasL)
thetaH_grad = T.grad(cost, thetaH)
biasH_grad = T.grad(cost, biasH)
theta1_grad = T.grad(cost, theta1)
bias1_grad = T.grad(cost, bias1)

gradient_step = theano.function(
    inputs=[X, y],
    updates=((thetaL, thetaL - ALPHA * thetaL_grad),
             (thetaH, thetaH - ALPHA * thetaH_grad),
             (theta1, theta1 - ALPHA * theta1_grad),
             (biasL, biasL - ALPHA * biasL_grad),
             (biasH, biasH - ALPHA * biasH_grad),
             (bias1, bias1 - ALPHA * bias1_grad)))


def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.show()
    # plt.scatter(y_train[:, 0], X_train[:, 1], c=y_train,
    # cmap=plt.cm.Spectral)


def step_gradient(batch):
    gradient_step(batch)


def def_batches(X_train=X_train, y_train=y_train, BATCH_SIZE=BATCH_SIZE):
    batch = np.random.permutation(y_train.shape[0])
    X_train = X_train[batch, :]
    y_train = y_train[batch]
    X_batches = np.array_split(X_train, BATCH_SIZE)
    y_batches = np.array_split(y_train, BATCH_SIZE)
    return X_batches, y_batches


def train_model(num_epochs=1000, print_cost=False):
    np.random.seed(0)
    min_cost = 10000000
    X_batches, y_batches = def_batches()
    print("running")
    for i in range(0, int(num_epochs*10)):
        bias1.set_value(np.random.randn(HIDDEN_DIM) /
                        np.sqrt(HIDDEN_DIM))
        biasH.set_value(np.random.randn(HIDDEN_DIM, (NUM_LAYERS - 2)) /
                        np.sqrt(HIDDEN_DIM))
        biasL.set_value(np.random.randn(OUTPUT_DIM) /
                        np.sqrt(OUTPUT_DIM))
        theta1.set_value(np.random.randn(INPUT_DIM, HIDDEN_DIM) /
                         np.sqrt(INPUT_DIM))
        thetaH.set_value(np.random.randn(HIDDEN_DIM, HIDDEN_DIM,
                         (NUM_LAYERS - 2)) / np.sqrt(HIDDEN_DIM))
        thetaL.set_value(np.random.randn(HIDDEN_DIM, OUTPUT_DIM) /
                         np.sqrt(HIDDEN_DIM))
        batch = np.random.randint(0, len(y_batches))
        cost = calculate_cost(X_batches[batch], y_batches[batch])
        if cost < min_cost:
            bias1_init = bias1.get_value()
            biasH_init = biasH.get_value()
            biasL_init = biasL.get_value()
            theta1_init = theta1.get_value()
            thetaH_init = thetaH.get_value()
            thetaL_init = thetaL.get_value()
            min_cost = cost
        if print_cost and i > 2 and i % 9 == 0:
            bias1.set_value(bias1_init)
            biasH.set_value(biasH_init)
            biasL.set_value(biasL_init)
            theta1.set_value(theta1_init)
            thetaH.set_value(thetaH_init)
            thetaL.set_value(thetaL_init)
            min_cost_full = calculate_cost(X_train, y_train)
            print("Min cost after random initialization {}: {}"
                  .format(i, min_cost_full))
    bias1.set_value(bias1_init)
    biasH.set_value(biasH_init)
    biasL.set_value(biasL_init)
    theta1.set_value(theta1_init)
    thetaH.set_value(thetaH_init)
    thetaL.set_value(thetaL_init)
    num_batches = len(y_batches)
    for i in range(0, num_epochs):
        for j in range(0, num_batches):
            step_gradient(X_batches[j], y_batches[j])
        if print_cost and i % 100 == 0:
            print("Cost after descent epoch {}: {}"
                  .format(i, calculate_cost(X_train, y_train)))

train_model(print_cost=True)
plot_decision_boundary(lambda x: predict(x))
