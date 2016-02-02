import theano
import theano.tensor as T
import numpy as np

A = np.zeros((2, 4, 3, 1)).astype('float32')
B = np.ones((2, 1, 3, 3)).astype('float32')

A_t = T.tensor4()
B_t = T.tensor4()

A_f = A_t.flatten(2)
B_f = B_t.flatten(2)

C = T.concatenate([A_f, B_f], axis=-1)

test = theano.function([A_t, B_t], C)

result = test(A, B)
print(result)
print(result.shape)
