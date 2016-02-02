# import OkapiV2.Core as ok
from OkapiV2 import Datasets, Optimizers, Activations, Losses
from OkapiV2.Core import Model, Branch
from OkapiV2.Layers.Basic import FullyConnected
from OkapiV2.Layers.Activations import ActivationLayer, PReLULayer
from OkapiV2.Layers.Convolutional import Convolutional, MaxPooling
import numpy as np
import pickle
from matplotlib import pyplot

X_train, y_train, X_val, y_val, X_test, y_test = Datasets.load_mnist()
X_obs, y_obs = X_train, y_train
X_reward = []
X_reward.append(np.zeros((X_obs.shape[0] * 10, X_obs.shape[1], X_obs.shape[2], X_obs.shape[3])))
X_reward.append(np.zeros((X_obs.shape[0] * 10, y_obs.shape[1])))
y_reward = np.zeros((X_obs.shape[0] * 10, 1))

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
tree.add_input(X_reward[0])
tree.add_input(X_reward[1])

model = Model()
model.set_tree(tree)
model.set_loss(Losses.MeanSquared())
model.add_output(y_reward)
learning_rate = 0.00002
model.set_optimizer(Optimizers.RMSprop(learning_rate=learning_rate))

model.compile(X_reward, y_reward, initialize_params=True)
file = open('reward_model_params_vec.pk', 'rb')
params = pickle.load(file)
file.close()
model.set_params_as_vec(params)
model.compile(X_reward, y_reward, initialize_params=False)

'''ok.save_model(model, 'okapi_reward_model.pk')
model = ok.load_model('okapi_reward_model.pk')'''
# model.set_dream_optimizer(Optimizers.RMSprop(learning_rate=0.0001, momentum=0.99))

preds = model.predict_dream([None, np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])], [(28, 28)], max_dream_length=100)
# print(model.predict([preds[0], np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])]))
pyplot.imshow(np.reshape(preds[0], (28, 28)), interpolation='nearest', cmap='Greys')
# pyplot.imshow(np.reshape(X_train[[0]], (28, 28)), interpolation='nearest', cmap='Greys')
pyplot.show()
'''reward = model.predict([X_train[[0]], np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])])
print(reward)
preds = model.predict_dream([X_train[[0]], None], [(10,)])
print(preds[0])'''
