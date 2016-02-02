import Okapi.Core as ok
from Okapi.Core import Model
from Okapi.Layers import FullyConnectedLayer, ActivationLayer, \
    GRULayer, DropoutLayer, BatchNormalizationLayer, PReLULayer
import Okapi.Activations as Activations
from Okapi.Optimizers import RMSprop
from Okapi.Losses import CategoricalCrossentropy
from Okapi.Initializers import RandNormalInit
import numpy as np
import time
import sys
import random
'''from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import GRU
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop as k_RMSprop'''

path = "data/shakespeare.txt"
print("Reading data...")
text = open(path).read()[0:1000]
print('Corpus length:', len(text))

chars = set(text)
print('Total characters:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 100
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))

print('Vectorizing data...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

X_train, y_train = X, y

batch_size = 128
h_layer_size = 50
d_layer_size = 20
learning_rate = 0.01
l1_param = 0.01
l2_param = 0.01
dropout_p = 0.5
max_num_layers = 10
max_layer_size = 50
rmsprop_momentum = 0.9

num_epochs = 5


start_time_1 = time.clock()
model = Model()
model.add(GRULayer((h_layer_size, 1, 1, 1), return_sequences=False))
model.add(BatchNormalizationLayer())
model.add(PReLULayer())
model.add(DropoutLayer(dropout_p))
model.add(GRULayer((h_layer_size, 1, 1, 1), return_sequences=False))
model.add(BatchNormalizationLayer())
model.add(PReLULayer())
model.add(DropoutLayer(dropout_p))
model.add(FullyConnectedLayer((d_layer_size, 1, 1, 1)))
model.add(BatchNormalizationLayer())
model.add(PReLULayer())
model.add(DropoutLayer(dropout_p))
model.add(FullyConnectedLayer())
model.add(ActivationLayer(Activations.softmax))

model.set_optimizer(RMSprop(learning_rate=learning_rate,
                    momentum=rmsprop_momentum))
model.set_loss(CategoricalCrossentropy(l1_param, l2_param))
model.set_initializer(RandNormalInit())
model.set_batch_size(batch_size)


def predict():
    def sample(a, temperature=1.0):
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('Diversity: ', diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
        for iteration in range(40):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            preds = model.predict(x)[0]
            try:
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]
                generated += next_char
                sentence = sentence[1:] + next_char
                sys.stdout.write(next_char)
                sys.stdout.flush()
            except ValueError:
                print("Value Error")

model.train(X_train, y_train, num_epochs=num_epochs,
            epoch_callback=predict)
end_time_1 = time.clock()
t1 = end_time_1 - start_time_1

ok.save_model(model)
del model
model = ok.load_model()

model.reinforce(X_train, y_train, num_epochs=num_epochs,
                epoch_callback=predict)

'''start_time_2 = time.clock()
model_keras = Sequential()
model_keras.add(GRU(h_layer_size, input_dim=len(chars),
                init='normal', return_sequences=False))
model_keras.add(BatchNormalization(mode=1))
model_keras.add(PReLU())
model_keras.add(Dropout(dropout_p))
model_keras.add(Dense(d_layer_size, init='normal'))
model_keras.add(BatchNormalization(mode=1))
model_keras.add(PReLU())
model_keras.add(Dropout(dropout_p))
model_keras.add(Dense(len(chars), init='normal'))
model_keras.add(BatchNormalization(mode=1))
model_keras.add(PReLU())
model_keras.add(Dropout(dropout_p))
model_keras.add(Activation('softmax'))

sgd = k_RMSprop(lr=learning_rate, rho=rmsprop_momentum)

model_keras.compile(loss='categorical_crossentropy', optimizer=sgd)
model_keras.fit(X_train, y_train, batch_size=batch_size, nb_epoch=num_epochs)
end_time_2 = time.clock()
t2 = end_time_2 - start_time_2

print("Okapi took {} seconds, and Keras took {} seconds".format(t1, t2))
if t1 < t2:
    print("Okapi was {} % faster than Keras!"
          .format(100 * (t2 - t1) / t2))
else:
    print("Okapi was {} % slower than Keras."
          .format(100 * (t1 - t2) / t2))'''
