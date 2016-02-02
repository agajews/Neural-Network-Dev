from OkapiV2.Core import Model
from OkapiV2.Layers.Basic import FullyConnected
from OkapiV2.Layers.Activations import ActivationLayer
from OkapiV2 import Activations
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD as k_SGD
from sklearn.datasets import make_multilabel_classification as multilabel
import time

batch_size = 256
num_classes = 20
num_features = 50
num_samples = 50000
h_layer_size = 500
num_epochs = 20
learning_rate = 0.001
reg_param = 0.001

X_train, y_train = multilabel(n_samples=num_samples,
                              n_features=num_features,
                              n_classes=num_classes)
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')

model = Model()
model.add(FullyConnected((h_layer_size, 1, 1, 1)))
model.add(ActivationLayer(Activations.tanh))
model.add(FullyConnected())
model.add(ActivationLayer(Activations.alt_softmax))

start_time_1 = time.time()
model.train(X_train, y_train, batch_size=batch_size, num_epochs=num_epochs)
end_time_1 = time.time()
t1 = end_time_1 - start_time_1

model_keras = Sequential()
model_keras.add(Dense(h_layer_size, input_dim=num_features, init='uniform'))
model_keras.add(Activation('tanh'))
model_keras.add(Dense(num_classes, init='uniform'))
model_keras.add(Activation('softmax'))

sgd = k_SGD(lr=learning_rate, nesterov=False)

start_time_2 = time.time()
model_keras.compile(loss='categorical_crossentropy', optimizer=sgd)
model_keras.fit(X_train, y_train, batch_size=batch_size, nb_epoch=num_epochs)
end_time_2 = time.time()
t2 = end_time_2 - start_time_2

print("Okapi took {} seconds, and Keras took {} seconds".format(t1, t2))
if t1 < t2:
    print("Okapi was {} % faster than Keras!"
          .format(100 * (t2 - t1) / t2))
else:
    print("Okapi was {} % slower than Keras."
          .format(100 * (t1 - t2) / t2))
