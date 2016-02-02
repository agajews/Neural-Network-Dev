import numpy as np


def get_fans(dims):
    fan_in = dims[0] if len(dims) == 2 else np.prod(dims[1:])
    fan_out = dims[1] if len(dims) == 2 else dims[0]
    return fan_in, fan_out


def uniform(dims, scale=0.05):
    range = (np.sqrt(3) * scale, -np.sqrt(3) * scale)
    return np.random.uniform(range[0], range[1], dims)


def zeros(dims):
    return np.zeros(dims)


def ones(dims):
    return np.ones(dims)


def normal(dims, scale=0.5):
    return np.random.standard_normal(dims) * scale


def glorot_uniform(dims):
    fan_in, fan_out = get_fans(dims)
    std = np.sqrt(6.0 / (fan_in + fan_out))
    return uniform(dims, std)


def glorot_normal(dims):
    fan_in, fan_out = get_fans(dims)
    std = np.sqrt(6.0 / (fan_in + fan_out))
    return normal(dims, std)


def lecun_uniform(dims):
    ''' Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    '''
    fan_in, fan_out = get_fans(dims)
    scale = np.sqrt(3. / fan_in)
    return uniform(dims, scale)


def he_normal(dims):
    ''' Reference:  He et al., http://arxiv.org/abs/1502.01852
    '''
    fan_in, fan_out = get_fans(dims)
    s = np.sqrt(2. / fan_in)
    return normal(dims, s)


def he_uniform(dims):
    fan_in, fan_out = get_fans(dims)
    s = np.sqrt(6. / fan_in)
    return uniform(dims, s)


def orthogonal(shape, scale=1.1):
    ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return scale * q[:shape[0], :shape[1]]
