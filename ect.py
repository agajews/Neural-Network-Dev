import OkapiV2.Core as ok
from OkapiV2.Core import Model
from OkapiV2.Layers.Basic import FullyConnected, Dropout
from OkapiV2.Layers.Activations import ActivationLayer, PReLULayer
from OkapiV2.Layers.Convolutional import Convolutional, MaxPooling
from OkapiV2 import Activations, Datasets, Initializers
import numpy as np
import random
from operator import itemgetter
from itertools import zip_longest as izip

X_train, y_train, X_val, y_val, X_test, y_test = Datasets.load_mnist()

dropout_p = 0.2
num_filters = 1
filter_size = 5
pool_size = 2
num_classes = 10
pad = False
batch_size = 1000

population_size = 100
num_generations = 100
rm_p = 0.8
init_mut_p = 1e-5
init_mut_std = 1e-3
init_cross_p = 0.7

model = Model()
model.add(Convolutional(num_filters, filter_size, filter_size, pad=pad))
model.add(ActivationLayer(Activations.tanh))
model.add(MaxPooling(pool_size, pool_size))
model.add(ActivationLayer(Activations.ReLU))
model.add(Dropout(dropout_p))
model.add(FullyConnected())
model.add(ActivationLayer(Activations.alt_softmax))

model.compile(X_train, y_train)

X_batches, y_batches, num_batches = \
    ok.make_batches(X_train, y_train, batch_size)

def get_loss(X_batch=None, y_batch=None, full=False):
    if full:
        loss = 0
        for X_batch, y_batch in izip(X_batches, y_batches):
            loss += model.get_train_loss(X_batch, y_batch)
        loss /= num_batches
    else:
        if X_batch is None or y_batch is None:
            ind = random.randrange(0, num_batches)
            X_batch, y_batch = X_batches[ind], y_batches[ind]
        loss = model.get_train_loss(X_batch, y_batch)
    return loss

step = 0.01

print('Starting training...')
current_loss = get_loss()
print('Init loss: {}'.format(current_loss))
params = model.get_params_as_vec()
for gen in range(num_generations):
    for i in range(params.shape[0]):
        ind = random.randrange(0, num_batches)
        X_batch, y_batch = X_batches[ind], y_batches[ind]

        prev_loss = current_loss
        params[i] += step
        model.set_params_as_vec(params)
        current_loss = get_loss(X_batch, y_batch)
        while current_loss < prev_loss:
            prev_loss = current_loss
            params[i] += step
            model.set_params_as_vec(params)
            current_loss = get_loss(X_batch, y_batch)
        print('Param {}/{}: {}'.format(i, params.shape[0], current_loss))
    print('Gen {}/{}: {}'.format(gen + 1, num_generations, current_loss))
    step *= -0.5

