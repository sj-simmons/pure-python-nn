# nndicts.py                                                   Simmons June 2018
"""
This provides two dictionaries, one whose targets are various activation
functions and their derivatives, and another whose targets are common loss
functions their derivatives (up to a constant).
"""

from math import exp
from operator import sub
from collections import namedtuple

##################  Some activations and their derivatives  ####################
#
# Sets an activation function f(x) and provides it's derivative as a function of
# y=f(x).
#
# Note: It's efficient to compute the derivative of sigmoid when it's express-
# ed as a function of y = sigmoid(x).  To make our code in the neural net
# classes nicer, we write all the derivatives below as function of the output
# of the corresponding activation function.  So if f(x) is the activation
# function in question, the derivative below is actually (d/dx)(f(x)) but
# written as a function of y = f(x).  This is what we denote by f'(y).
# Now suppose that we want to take the derivative of g(f(x)) with respect to
# x.  Using the chain rule we get: (d/dy)(g(y)) * (d/dx)(f(x)) = g'(y)*f'(y)
# where y = f(x).

def kron_d(i, j):
  """ The Kronecker delta. """
  return 1 if i == j else 0

def scal_div(value, lst):
  """ Return lst with each element divided by value. """
  return [elt/value for elt in lst]

ACTIVATIONS = {
    'id': namedtuple('id', 'func der')(
        lambda x: x,
        lambda y: 1
    ),
    'sigmoid': namedtuple('sigmoid', 'func der')(
        lambda x: 1 / (1 + exp(-x)),
        lambda y: y * (1 - y)
    ),
    'ReLU': namedtuple('ReLU', 'func der')(
        lambda x: max(0, x),
        lambda y: 0 if y == 0 else 1
    ),
    'softmax': namedtuple('softmax', 'func der')(
        lambda xs: scal_div(sum(map(exp, xs)), [exp(x) for x in xs]),
        lambda ys, k: [ys[k]*(kron_d(j, k) - ys[j]) for j, _ in enumerate(ys)]
    )
}

ACTIVATIONS[None] = ACTIVATIONS['id']


###################  Loss functions and their derivatives  #####################
#
# Sets a loss function L(y_hat) = L(y_hat, y) where y_hat and y are vectors
# (meaning lists). Also provides L'(y_hat) = dL(y_hat)/dy_hat
#
# Note: For a batch or minibatch, L is one summand of the total Loss function.

LOSS_FUNCTIONS = {
    'MSE': namedtuple('MSE', 'func der')(  # mean squared error
        lambda y_hat, y: map(lambda x: x**2, map(sub, y_hat, y)),
        lambda y_hat, y: map(sub, y_hat, y)
    )
#    'NLL': namedtuple('NLL', 'func der')(  # negative log likelihood
#        lambda y_hat, y: (y_hat - y)**2,
#        lambda y_hat, y: y_hat - y
#    )
}


def test():
  """
  >>> sigmoid = ACTIVATIONS['sigmoid'].func
  >>> sigmoid(0)
  0.5
  >>> softmax = ACTIVATIONS['softmax'].func
  >>> softmax([0,1]) == [1/(1+exp(1)), 1/(1+exp(-1))]
  True

  #>>> den = sigmoid(1) + sigmoid(2) + sigmoid(3)
  #>>> sigmoid(1)/den; sigmoid(2)/den; sigmoid(3)/den
  #
  #>>> softmax([1,2,3],0); softmax([1,2,3],1); softmax([1,2,3],2);

  >>> activations = map(lambda s: ACTIVATIONS[s], [None,'id','ReLU','sigmoid'])
  >>> [acitivation.func(-7) for acitivation in activations] # doctest: +ELLIPSIS
  [-7, -7, 0, 0.000911...

  # The code above works but activations is a map object that is lazy evaluated,
  # so we cannot get our hands on, for instance, the first activation; so that,
  # for example,
  #
  >>> list(activations)[0]
  Traceback (most recent call last):
      ...
  IndexError: list index out of range

  # For our application below, do can instead use a list comprehension:
  #
  >>> activations = [ACTIVATIONS[str_] for str_ in ['id', 'ReLU', 'sigmoid']]
  >>> activations[0]  # doctest: +ELLIPSIS
  id(func=<function <lambda>...
  >>> activations[1].func(-3)
  0
  """

if __name__ == '__main__':
  import doctest
  doctest.testmod()
