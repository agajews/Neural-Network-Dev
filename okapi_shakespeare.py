import numpy as np
import random
import sys
import time

maxlen = 100
step = 3
num_chars = 400
diversities = [0.2, 0.5]

batch_size = 128
h_layer_size = 512
learning_rate = 0.001
momentum = 0.99
num_iterations = 36
corpus_length = 100000


def main_okapi():
    import OkapiV2.Core as ok
    from OkapiV2.Core import Model
    from OkapiV2.Layers.Basic import FullyConnected, Dropout, BatchNorm
    from OkapiV2.Layers.Activations import ActivationLayer, PReLULayer
    from OkapiV2.Layers.Recurrent import LSTM
    from OkapiV2 import Activations, Optimizers, Losses

    path = 'data/lear.txt'
    text = open(path).read().lower()  # [0:corpus_length]
    print('Corpus length:', len(text))

    chars = set(text)
    print('Total Characters:', len(chars))
    char_to_index = dict((c, i) for i, c in enumerate(chars))
    index_to_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('Total Sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_to_index[char]] = 1
        y[i, char_to_index[next_chars[i]]] = 1

    def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a)) - 1e-7
        return np.argmax(np.random.multinomial(1, a, 1))

    model = Model()
    model.add(LSTM((h_layer_size, 1, 1, 1)))
    model.add(PReLULayer())
    model.add(Dropout(0.2))
    model.add(BatchNorm())
    model.add(LSTM((h_layer_size, 1, 1, 1)))
    model.add(PReLULayer())
    model.add(Dropout(0.2))
    model.add(BatchNorm())
    model.add(FullyConnected())
    model.add(ActivationLayer(Activations.alt_softmax))

    model.set_loss(Losses.Crossentropy())
    model.set_optimizer(Optimizers.RMSprop(learning_rate=learning_rate))

    for iteration in range(0, num_iterations):
        print()
        print('-' * 50)
        print('Iteration', iteration + 1)
        model.train(X, y, batch_size=batch_size, num_epochs=1,
                    params_filename='okapi_shakespeare_params.pk')

        start_index = random.randint(0, len(text) - maxlen - 1)

        for diversity in diversities:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for iteration in range(num_chars):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_to_index[char]] = 1.

                preds = model.predict(x)
                preds = preds[0]
                next_index = sample(preds, diversity)
                next_char = index_to_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
    ok.save_model(model, 'okapi_shakespeare_model.pk')


def main_keras():
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Dropout
    from keras.layers.recurrent import LSTM
    # from keras.layers.advanced_activations import PReLU
    # from keras.layers.normalization import BatchNormalization
    from keras.optimizers import RMSprop

    path = 'data/lear.txt'
    text = open(path).read().lower()  # [0:corpus_length]
    print('Corpus length:', len(text))

    chars = set(text)
    print('Total Characters:', len(chars))
    char_to_index = dict((c, i) for i, c in enumerate(chars))
    index_to_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('Total Sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_to_index[char]] = 1
        y[i, char_to_index[next_chars[i]]] = 1

    def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))

    model = Sequential()
    model.add(LSTM(h_layer_size, return_sequences=True,
                   input_shape=(maxlen, len(chars))))
    model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    model.add(LSTM(h_layer_size, return_sequences=False))
    model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    rmsprop = RMSprop(lr=learning_rate, rho=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

    for iteration in range(0, num_iterations):
        print()
        print('-' * 50)
        print('Iteration', iteration + 1)
        model.fit(X, y, batch_size=batch_size, nb_epoch=1)

        start_index = random.randint(0, len(text) - maxlen - 1)

        for diversity in diversities:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for iteration in range(num_chars):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_to_index[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = index_to_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

okapi_start = time.clock()
main_okapi()
okapi_time = round(time.clock() - okapi_start, 2)

'''keras_start = time.clock()
main_keras()
keras_time = round(time.clock() - keras_start, 2)

print('Okapi Time: {} \nKeras Time: {}'
      .format(okapi_time, keras_time))'''
