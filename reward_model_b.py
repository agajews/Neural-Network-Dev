import theano
import theano.tensor as T
import pickle
import OkapiV2.Core as ok
from OkapiV2.Core import Model, Branch
from OkapiV2.Layers.Basic import FullyConnected
from OkapiV2.Layers.Activations import ActivationLayer, PReLULayer
from OkapiV2.Layers.Convolutional import Convolutional, MaxPooling
from OkapiV2 import Activations, Datasets, Losses, Optimizers, Initializers
import numpy as np

X_train, y_train, X_val, y_val, X_test, y_test = Datasets.load_mnist()
obs_index = 10000
X_obs, y_obs = X_train[:obs_index], y_train[:obs_index]
X_reward = []
X_reward.append(np.zeros((X_obs.shape[0] * 10, X_obs.shape[1], X_obs.shape[2], X_obs.shape[3])))
X_reward.append(np.zeros((X_obs.shape[0] * 10, y_obs.shape[1])))
y_reward = np.zeros((X_obs.shape[0] * 10, 1))

print('Compiling reward...')
y_hat = T.matrix(dtype='float32')
y = T.matrix(dtype='float32')
preds = T.argmax(y_hat, axis=1)
true = T.argmax(y, axis=1)
actual_reward = T.switch(T.eq(preds, true), 1,  0) - 0.0 * T.abs_(1 - T.sum(y_hat, axis=1))
reward = theano.function([y_hat, y], actual_reward)

print('Expanding data...')
for i in range(len(y_obs)):
    for j in range(10):
        row = i * 10 + j
        action = np.zeros((1, 10))
        action[:, j] = 1
        # action = Initializers.uniform((1, 10), 1)
        X_reward[0][row, :, :, :] = X_obs[i, :, :, :]
        X_reward[1][row, :] = action.flatten(1)
        y_reward[row, :] = reward(action.astype('float32'), y_obs[[i]].astype('float32'))
        if (row + 1) % 10000 is 0:
            print('{}/{}: {}'.format(row + 1, y_reward.shape[0], y_reward[row, :]))

dropout_p = 0.2
num_filters = 32
filter_size = 3
pool_size = 2
num_classes = 10
pad = True

state_branch = Branch()
state_branch.add_layer(Convolutional(num_filters, filter_size, filter_size, pad=pad))
state_branch.add_layer(PReLULayer())
state_branch.add_layer(MaxPooling(pool_size, pool_size))
state_branch.add_layer(Convolutional(num_filters, filter_size, filter_size, pad=pad))
state_branch.add_layer(PReLULayer())
state_branch.add_layer(MaxPooling(pool_size, pool_size))
state_branch.add_layer(FullyConnected((100, 1, 1, 1)))
state_branch.add_layer(ActivationLayer(Activations.tanh))
state_branch.add_input(X_reward[0])

action_branch = Branch()
action_branch.add_layer(FullyConnected((100, 1, 1, 1)))
action_branch.add_layer(PReLULayer())
action_branch.add_input(X_reward[1])

tree = Branch()
tree.add_layer(FullyConnected((512, 1, 1, 1)))
tree.add_layer(PReLULayer())
tree.add_layer(FullyConnected((512, 1, 1, 1)))
tree.add_layer(PReLULayer())
tree.add_layer(FullyConnected((512, 1, 1, 1)))
tree.add_layer(PReLULayer())
tree.add_layer(FullyConnected())
tree.add_layer(ActivationLayer(Activations.tanh))
tree.add_input(X_reward[0])
tree.add_input(X_reward[1])

model = Model()
model.set_tree(tree)
model.set_loss(Losses.MeanSquared())
learning_rate = 0.00002
model.set_optimizer(Optimizers.RMSprop(learning_rate=learning_rate))
# model.compile(X_reward, y_reward)
# model.train(X_reward, y_reward, 24)

reinforce_index = X_obs.shape[0]
X_batches, y_batches, num_batches = ok.make_batches([X_train], y_train, batch_size=10000)
for i in range(8):
    print('\n---Iteration {}---'.format(i + 1))
    for X_batch, y_batch in zip(X_batches, y_batches):
        model.train(X_reward, y_reward, 24)
        accuracy, preds = model.get_dream_accuracy([X_batch[0], None], y_batch)
        preds = preds[0]
        preds += Initializers.normal(preds.shape, 0.01)
        print('Accuracy: {}%'.format(accuracy))
        preds_reward = reward(preds.reshape(preds.shape[0], preds.shape[1]).astype('float32'),
                y_batch.reshape(preds.shape[0], preds.shape[1]).astype('float32'))
        print('Avg Reward: {}'.format(np.mean(preds_reward)))
        X_reward[0] = np.append(X_reward[0], X_batch[0], axis=0)
        X_reward[1] = np.append(X_reward[1], preds, axis=0)
        y_reward = np.append(y_reward, preds_reward.reshape(preds_reward.shape[0], 1), axis=0)
        params = model.get_params_as_vec()
        file = open('reward_model_params_vec.pk', 'wb')
        pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    test_accuracy, test_preds = model.get_dream_accuracy([X_test, None], y_test)
    print('Test Accuracy: {}%'.format(test_accuracy))
