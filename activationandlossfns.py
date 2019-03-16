# activationandlossfns.py                                      Simmons June 2018
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
# Note: It's more efficient to compute the derivative of the sigmoid function
# when it is expressed as a function of y, where y = sigmoid(x), rather than of
# a fuction of x.  To make our code in the neural net classes nicer, we similar-
# ly write all the derivatives below as function of the output of the corres-
# ponding activation function.
#
# So if f(x) is the activation function in question, the derivative below is
# actually (d/dx)(f(x)) but written as a function of y = f(x).  This is what we
# denote by f'(y).
#
# Now suppose that we want to take the derivative of g(f(x)) with respect to x.
# Using the chain rule we get: (d/dy)(g(y)) * (d/dx)(f(x)) = g'(y)*f'(y)
# where y = f(x).
#
# Note2: Any of these that will be used as an activation for the output layer
# should output a list, not just a number.

def kron_d(i, j):
  """ The Kronecker delta. """
  return 1 if i == j else 0

ACTIVATIONS = {
    'id': namedtuple('id', 'func der')(
        lambda x: x,
        lambda y: [1]
    ),
    'relu': namedtuple('ReLU', 'func der')(
        lambda x: max(0, x),
        lambda y: [0] if y == 0 else [1]
    ),
    'sigmoid': namedtuple('sigmoid', 'func der')(
        lambda x: 1 / (1 + exp(-x)),
        lambda ys: [y * (1 - y) for y in ys]
    ),
    'softmax': namedtuple('softmax', 'func der')(
        lambda xs, x: exp(x)/sum(map(lambda z: exp(z), xs)),
        lambda ys: [y * (1 - y) for y in ys]
    )
}

ACTIVATIONS[None] = ACTIVATIONS['id']

###################  Loss functions and their derivatives  #####################
#
# Sets a loss function J(y_hats) = J(y_hats, ys) where y_hats and ys are vectors
# (meaning lists). Also provides J'(y_hats) = dJ(y_hats)/dy_hats. Both the J and
# J' return lists so as to handle nets with multiple outputs.
#
# Note: For a batch or minibatch, J is one summand of the total loss function.

LOSS_FUNCTIONS = {
    'MSE': namedtuple('MSE', 'func der')(  # mean squared error
        lambda y_hats, ys: map(lambda x: x**2, map(sub, y_hats, ys)),
        lambda y_hats, ys: map(sub, y_hats, ys)
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
  >>> dsigmoid = ACTIVATIONS['sigmoid'].der
  >>> dsigmoid([sigmoid(1)]) == [sigmoid(1)*(1-sigmoid(1))]
  True
  >>> softmax = ACTIVATIONS['softmax'].func
  >>> softmax([0,1], 0) == 1/(1+exp(1))
  True
  >>> softmax([0,1], 1) == 1/(1+exp(-1))
  True

  >>> activations = map(lambda s: ACTIVATIONS[s], [None,'id','ReLU','sigmoid'])
  >>> [acitivation.func(-7) for acitivation in activations] # doctest: +ELLIPSIS
  [-7, -7, 0, 0.000911...

  # The code above works but activations is a map object that is lazy evaluated.
  # Hence we cannot get our hands on, for instance, the first activation; so
  # that, for example,
  >>> list(activations)[0]
  Traceback (most recent call last):
      ...
  IndexError: list index out of range

  # You do can instead use a list comprehension:
  >>> activations = [ACTIVATIONS[str_] for str_ in ['id', 'ReLU', 'sigmoid']]
  >>> activations[0]  # doctest: +ELLIPSIS
  id(func=<function <lambda>...
  >>> activations[1].func(-3)
  0
  """

if __name__ == '__main__':
  import doctest
  doctest.testmod()
