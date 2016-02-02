import nltk
import numpy as np
import csv
import itertools


def read_data(input_file, sentence_position, vocabulary_size=10000,
              unknown_token="_UNKNOWN_TOKEN",
              sentence_start_token="_SENTENCE_START",
              sentence_end_token="_SENTENCE_END"):
    print("Reading input CSV...")
    with open(input_file, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        sentences = itertools.chain(*[nltk.sent_tokenize(x[sentence_position])
                                    for x in reader])
        sentences = ["%s %s %s" % (sentence_start_token, x,
                                   sentence_end_token) for x in sentences]
        print("Parsed {} sentences.".format(len(sentences)))
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found {} unique words tokens.".format(len(word_freq.items())))
        vocab = word_freq.most_common(vocabulary_size-1)
        index_to_word = [x[sentence_position] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
        print("Using vocabulary size {}.".format(vocabulary_size))
        print("The least frequent word in our vocabulary is '{}' \
              and appeared {} times.".format(vocab[-1][0], vocab[-1][1]))
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index
                                      else unknown_token for w in sent]
        print("\nExample sentence: '{}'".format(sentences[0]))
        print("\nProcessed Sentence: '{}'".format(tokenized_sentences[0]))
        X_train = np.asarray([[word_to_index[w]
                              for w in sent[:-1]]
                              for sent in tokenized_sentences])
        y_train = np.asarray([[word_to_index[w]
                              for w in sent[1:]]
                              for sent in tokenized_sentences])
        return X_train, y_train, index_to_word, word_to_index


class RNNNumpy:

    def __init__(self, X_train, y_train, hidden_dim=100, bptt_truncation=4):
        self.vocabulary_size = X_train.shape[0]
        print(self.vocabulary_size)
        self.hidden_dim = hidden_dim
        self.bptt_truncation = bptt_truncation
        self.U = np.random.uniform(-np.sqrt(1./self.vocabulary_size),
                                   np.sqrt(1./self.vocabulary_size),
                                   (hidden_dim, self.vocabulary_size))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim),
                                   np.sqrt(1./hidden_dim),
                                   (self.vocabulary_size, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim),
                                   np.sqrt(1./hidden_dim),
                                   (hidden_dim, hidden_dim))

    def forward_prop(self, x):
        T = len(x)
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((T, self.vocabulary_size))
        for t in np.arange(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = self.softmax(self.V.dot(s[t]))
        return o, s

    def calculate_cost(self, x, y):
        J = 0
        for i in np.arange(len(y)):
            o, s = self.forward_prop(x[i])
            correct_predictions = o[np.arange(len(y[i])), y[i]]
            J += -1 * np.sum(np.log(correct_predictions))
        N = np.sum((len(y_i) for y_i in y))
        return J / N

    def bptt(self, x, y):
        T = len(y)
        o, s = self.forward_propagation(x)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            for bptt_step in np.arange(max(0, t-self.bptt_truncate),
                                       t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:, x[bptt_step]] += delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_descent(self, x, y, alpha):
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.U -= alpha * dLdU
        self.V -= alpha * dLdV
        self.W -= alpha * dLdW

    def train_net(self, alpha, num_iterations, cost_eval_spacing=5):
        costs = []


    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def softmax(self, A):
        d = np.exp(A)
        return d / sum(d)


'''X_train, y_train, index_to_word, word_to_index \
    = read_data('data/reddit-comments-2015-08.csv', 0)

np.random.seed(10)
net = RNNNumpy(X_train)
o, s = net.forward_prop(X_train[10])
print(o.shape)
print(o)'''
