import numpy as np
from urllib.request import urlretrieve
import gzip
import os
import tarfile
import sys
import pickle
import random


def get_file(filename, source, offset=16):
    download(filename, source)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=offset)
    return data


def download(filename, source):
    if not os.path.exists(filename):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)


def untar(filename):
    file = tarfile.open(filename)
    file.extractall()
    file.close()


def vec_to_onehot(vec):
    num_categories = len(np.unique(vec))
    vec_placeholder = np.zeros((vec.shape[0], num_categories),
                               dtype='float32')
    vec_placeholder[np.arange(vec.shape[0]), vec] = 1
    return vec_placeholder


def get_val_set(X_train, y_train, val_size):
    X_train, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]
    return X_train, y_train, X_val, y_val


def load_mnist(val_size=10000):
    print("Loading data...")
    source = 'http://yann.lecun.com/exdb/mnist/'

    def load_mnist_images(filename):
        data = get_file(filename, source)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        return get_file(filename, source, offset=8)

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    y_train, y_test = vec_to_onehot(y_train), vec_to_onehot(y_test)
    X_train, y_train, X_val, y_val = get_val_set(X_train, y_train, val_size)

    return X_train, y_train, X_val, y_val, X_test, y_test


class TextData():
    def __init__(self, filename, num_examples=None, maxlen=40, stride=3):
        self.filename = filename
        self.num_examples = num_examples
        self.maxlen = maxlen
        self.stride = stride

    def get_data(self):
        print("Loading data...")
        text = open(self.filename).read()
        if self.num_examples is not None:
            text = text[0:self.num_examples]
        chars = set(text)
        char_to_index = dict((c, i) for i, c in enumerate(chars))
        index_to_char = dict((i, c) for i, c in enumerate(chars))
        sentences = []
        next_chars = []
        for i in range(0, len(text) - self.maxlen, self.stride):
            sentences.append(text[i: i + self.maxlen])
            next_chars.append(text[i + self.maxlen])
        X_train = np.zeros((len(sentences), self.maxlen, len(chars)),
                           dtype=np.bool)
        y_train = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X_train[i, t, char_to_index[char]] = 1
            y_train[i, char_to_index[next_chars[i]]] = 1
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char
        self.text = text
        self.chars = chars
        return X_train, y_train

    def sample(self, a, temperature=1.0):
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))

    def predict(self, model, num_chars=1000,
                diversities=[0.2, 0.5, 1.0, 1.2]):
        start_index = random.randint(0, len(self.text) - self.maxlen - 1)
        for diversity in diversities:
            print("Diversity: {}".format(diversity))
            generated = ""
            sentence = self.text[start_index:start_index + self.maxlen]
            print('Seed: "{}"'.format(sentence))
            sys.stdout.write(generated)
            for i in range(num_chars):
                X = np.zeros((1, self.maxlen, len(self.chars)))
                for t, char in enumerate(sentence):
                    X[0, t, self.char_to_index[char]] = 1
                preds = model.predict(X)[0]
                try:
                    next_index = self.sample(preds, diversity)
                    next_char = self.index_to_char[next_index]
                    generated += next_char
                    sentence = sentence[1:] + next_char
                    sys.stdout.write(next_char)
                    sys.stdout.flush()
                except ValueError:
                    print("Value Error")


def load_cifar10(val_size=10000):
    def load_batch(fpath, label_key='labels'):
        f = open(fpath, 'rb')
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding="bytes")
            for k, v in d.items():
                del(d[k])
                d[k.decode("utf8")] = v
        print("Loading data...")
        f.close()
        data = d["data"]
        labels = d[label_key]

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels

    filename = 'cifar-10-python.tar.gz'
    download(filename, 'http://www.cs.toronto.edu/~kriz/')
    untar(filename)
    path = 'cifar-10-batches-py'

    num_train_examples = 50000

    X_train = np.zeros((num_train_examples, 3, 32, 32), dtype="uint8")
    y_train = np.zeros((num_train_examples,), dtype="uint8")

    for i in range(1, 6):
        filepath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(filepath)
        X_train[(i-1)*10000:i*10000, :, :, :] = data
        y_train[(i-1)*10000:i*10000] = labels

    filepath = os.path.join(path, 'test_batch')
    X_test, y_test = load_batch(filepath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    y_train, y_test = vec_to_onehot(y_train), vec_to_onehot(y_test)
    X_train, y_train, X_val, y_val = get_val_set(X_train, y_train, val_size)

    return X_train, y_train, X_val, y_val, X_test, y_test
