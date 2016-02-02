import numpy as np
import sklearn.datasets

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)

INPUT_DIM = 2
OUTPUT_DIM = 2
ALPHA = 0.01
LAMBDA = 0.01


def calculate_cost(model, num_layers, X=X, y=y, LAMBDA=LAMBDA):
    theta = list(range(0, num_layers))
    bias = list(range(0, num_layers))
    sum = list(range(0, num_layers))
    layer = list(range(0, num_layers))
    for i in range(1, num_layers-1):
        theta[i] = model["theta{}".format(i)]
        bias[i] = model["bias{}".format(i)]
    num_examples = len(X)
    layer[0] = X
    for i in range(1, num_layers-1):
        sum[i] = layer[i-1].dot(theta[i] + bias[i])
        layer[i] = np.tanh(sum[i])
    sum[num_layers-1] = layer[i-1].dot(theta[i] + bias[i])
    layer[num_layers-1] = np.exp(sum[num_layers-1])
    probs = layer[num_layers-1] / np.sum(layer[num_layers-1], axis=1,
                                         keepdims=True)
    corect_logprobs = -np.log(probs[range(num_examples), y])
    cost = np.sum(corect_logprobs)
    for i in range(1, num_layers-1):
        cost += LAMBDA/2 * (np.sum(np.square(theta[i])))
    return 1./num_examples * cost


def build_model(hidden_dim, num_layers=3, input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM, X=X, ALPHA=ALPHA, LAMBDA=LAMBDA,
                num_passes=2001, print_cost=False):
    num_examples = len(X)
    np.random.seed(0)
    '''
    theta1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
    bias1 = np.zeros((1, hidden_dim))
    theta2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
    bias2 = np.zeros((1, output_dim))
    '''
    theta = list(range(0, num_layers))
    bias = list(range(0, num_layers))

    theta[1] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
    bias[1] = np.zeros((1, hidden_dim))

    if num_layers > 3:
        for i in range(2, num_layers-2):
            theta[i] = np.random.randn(hidden_dim, hidden_dim)\
                / np.sqrt(hidden_dim)
            bias[i] = np.zeros((1, hidden_dim))

    theta[num_layers-1] = np.random.randn(hidden_dim, output_dim) \
        / np.sqrt(hidden_dim)
    bias[num_layers-1] = np.zeros((1, output_dim))

    model = {}

    for i in range(0, num_passes):
        sum = list(range(0, num_layers))
        layer = list(range(0, num_layers))
        delta = list(range(0, num_layers+1))
        d_theta = list(range(0, num_layers))
        d_bias = list(range(0, num_layers))

        layer[0] = X

        for i in range(1, num_layers-1):
            sum[i] = layer[i-1].dot(theta[i] + bias[i])
            layer[i] = np.tanh(sum[i])
        sum[num_layers-1] = layer[i-1].dot(theta[i] + bias[i])
        layer[num_layers-1] = np.exp(sum[num_layers-1])

        probs = layer[num_layers-1] / np.sum(layer[num_layers-1], axis=1,
                                             keepdims=True)

        delta[num_layers] = probs
        delta[num_layers][range(num_examples), y] -= 1
        for i in range(num_layers-1, 1, -1):
            print(i)
            delta[i] = delta[i+1].dot(theta[i].T) * \
                (1 - np.power(layer[i-1], 2))
        for i in range(2, num_layers):
            d_theta[i] = (layer[i-1].T).dot(delta[i+1])
            d_bias[i] = np.sum(delta[i+1], axis=0, keepdims=True)
        d_theta[1] = np.dot(X.T, delta[2])
        d_bias[1] = np.sum(delta[2], axis=0)

        for i in range(1, num_layers):
            theta[i] += -ALPHA * d_theta[i]
            bias[i] += -ALPHA * d_bias[i]
            model["theta{}".format(i)] = theta[i]
            model["bias{}".format(i)] = bias[i]

        '''
        sum1 = X.dot(theta1) + bias1
        layer1 = np.tanh(sum1)
        sum2 = layer1.dot(theta2) + bias2
        layer2 = np.exp(sum2)
        probs = layer2 / np.sum(layer2, axis=1, keepdims=True)

        delta3 = probs
        delta3[range(num_examples), y] -= 1
        d_theta2 = (layer1.T).dot(delta3)
        d_bias2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(theta2.T) * (1 - np.power(layer1, 2))
        d_theta1 = np.dot(X.T, delta2)
        d_bias1 = np.sum(delta2, axis=0)

        d_theta2 += LAMBDA * theta2
        d_theta1 += LAMBDA * theta1

        theta1 += -ALPHA * d_theta1
        bias1 += -ALPHA * d_bias1
        theta2 += -ALPHA * d_theta2
        bias2 += -ALPHA * d_bias2

        model = {'theta1': theta1, 'bias1': bias1, 'theta2': theta2,
                 'bias2': bias2}
        '''

        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}"
                  .format(i, calculate_cost(model, num_layers)))

model = build_model(3, 3, print_cost=True)
