from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
# from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys

'''
    Example script to generate text from Nietzsche's writings.
    At least 20 epochs are required before the generated text
    starts sounding coherent.
    It is recommended to run this script on GPU, as recurrent
    networks are quite computationally intensive.
    If you try this script on new data, make sure your corpus
    has at least ~100k characters. ~1M is better.
'''

'''path = get_file('nietzsche.txt',
            origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")'''
path = "data/shakespeare.txt"
print("Reading data...")
text = open(path).read()
print('Corpus length:', len(text))

chars = set(text)
print('Total characters:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

'''maxlen = 20
step = 3
paragraphs = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    paragraphs.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(paragraphs))'''

delimeter = "\n\n"
paragraphs = text.split(delimeter, text.count(delimeter))
paragraphs = [paragraph for paragraph in paragraphs if paragraph.strip()]
# paragraphs = filter(lambda name: name.strip(), paragraphs)
print("Number of paragraphs: {}".format(len(paragraphs)))

maxlen = 20
step = 3
sentences = []
next_chars = []
for paragraph in paragraphs:
    for i in range(0, len(paragraph) - maxlen, step):
        sentences.append(paragraph[i: i + maxlen])
        next_chars.append(paragraph[i + maxlen])
print('Number of sequences:', len(sentences))

X = np.zeros((len(sentences), maxlen, len(chars) + 1), dtype='float32')
X[:, :, -1] = np.random.uniform(-1, 1, (len(sentences), maxlen))
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen,
                                                        len(chars) + 1)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for iteration in range(400):
            x = np.zeros((1, maxlen, len(chars) + 1))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            x[:, :, -1] = np.random.uniform(-1, 1, (1, maxlen))

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
