import numpy as np


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
y = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)

syn0 = 2 * np.random.random((3, 1)) - 1

l1 = None
for i in xrange(10000):
    l0 = X
    dot_tmp = np.dot(l0, syn0)
    l1 = nonlin(dot_tmp)

    l1_error = y - l1
    tmp = nonlin(l1, True)
    l1_delta = l1_error * tmp

    tmp2 = np.dot(l0.T, l1_delta)
    syn0 += tmp2

print('Output after training:')
print(l1)
