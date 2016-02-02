import theano
import theano.tensor as T
import numpy as np
# from OkapiV2.Layers.Activations import SoftmaxLayer
from OkapiV2 import Losses

x, y = T.matrices('xy')

# regular softmax and crossentropy
sm = T.nnet.softmax(x)
cm1 = T.nnet.categorical_crossentropy(sm, y)
g1 = T.grad(cm1.mean(), x)

# numerically stable log-softmax with crossentropy
'''xdev = x-x.max(1, keepdims=True)
lsm = xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))'''
# lsm = SoftmaxLayer().get_output(x, None) + 1e-7
'''sm2 = T.exp(lsm)
cm2 = -T.sum(y*lsm, axis=1)'''
cm2 = Losses.AltSoftmaxLoss().get_train_loss(x, y, [None])
# cm2 = T.nnet.categorical_crossentropy(sm2, y)
g2 = T.grad(cm2.mean(), x)

# create some inputs into a softmax that are large and labels
a = np.exp(10*np.random.rand(5, 10).astype(theano.config.floatX))
# create some one-hot coded labels
b = np.eye(5, 10).astype(theano.config.floatX)

# show equivalence of softmax and exponentiated numerically stable log-softmax
'''f1 = theano.function([x], [sm, sm2])
sm1, sm2 = f1(a)
print(np.allclose(sm1, sm2))'''

# now show that the two versions result in the same crossentropy cost
f2 = theano.function([x, y], [cm1, cm2])
c1, c2 = f2(a, b)
print(np.allclose(c1, c2))

# now, show that in the standard softmax case the gradients blow up
# while in the log-softmax case they don't
f3 = theano.function([x, y], [g1, g2])
g1_, g2_ = f3(a, b)
print(g1_)
print(g2_)
