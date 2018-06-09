# ffnn.py                                                     Simmons  Spring 18
"""
This implements a feed-forward, fully-connected neural net in pure Python that
trains using stochastic gradient descent.
"""

import math
import random
from functools import reduce
from operator import add, and_
from collections import namedtuple

DEBUG_INST = False   # Print debugging info during instantiation of a Net.
DEBUG_TRAIN = False  # Print debugging info during training.


##################  Some activations and their derivatives  ####################

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

ACTIVATIONS = {
    'sigmoid': namedtuple('sigmoid', 'func der')(
        lambda x: 1 / (1 + math.exp(-x)),
        lambda y: y * (1 - y)
    ),
    'ReLU': namedtuple('ReLU', 'func der')(
        lambda x: max(0, x),
        lambda y: 0 if y == 0 else 1
    ),
    'id': namedtuple('id', 'func der')(lambda x: x, lambda y: 1),
    None: namedtuple('None_', 'func der')(lambda x: x, lambda y: 1)
}

# Notes on using the ACTIVATIONS dictionary above:
#
# Consider these two commands:
#
# > activations = map(lambda s: ACTIVATIONS[s], [None, 'id', 'ReLU', 'sigmoid'])
# > [acitivation.func(-7) for acitivation in activations] # doctest:+ELLIPSIS
# [-7, -7, 0, 0.000911...
#
# The code above works but activations is a map object that is lazy evaluated,
# so we cannot get our hands on, for instance, the first activation, so that,
# for example,
#
# > list(activations)[0]
#
# thows an exception with a list_index_out_of_range error.
#
# For our application below, do can this instead:
#
# > activations = [ACTIVATIONS[str_] for str_ in ['id', 'ReLU', 'sigmoid']]
# > activations[0]
# id(func=<function <lambda>...
# > activations[1].func(-3)
# 0


###################  Loss functions and their derivatives  #####################

# Sets a loss function L(y_hat) = L(y_hat, y) to be used with (mini-)batches of
# examples. Also provides L'(y_hat) = dL(y_hat)/dy_hat
#
# Note: For a minibatch, L is one summand of the Loss function applied to the
# mini-batch.

LOSS_FUNCTIONS = {
    'MSE': namedtuple('MSE', 'func der')(
        lambda y_hat, y: (y_hat - y)**2,
        lambda y_hat, y: y_hat - y
    ),
    'softmax': namedtuple('softmax', 'func der')(
        lambda y_hat, y: (y_hat - y)**2,
        lambda y_hat, y: y_hat - y
    )
}


#############################  The Neural Net  #################################

class _InputLink(object):
  """
  A connection from one node to another.

  Args:
    node (:obj:`_Node`): The node that this inputLink is to emantate from.
    weight (:obj: `Number`): This links initial weight.

  Attributes:
    input_node (:obj:`_Node`): The node that this inputLink emantates from.
    weight: (:obj:`Number`): This links weight.
    partial (:obj:`Number`): Holds the scaled partial w/r to this links
      weight.
  """
  def __init__(self, node, weight):
    if DEBUG_INST:
      print("    inputLink created")
    self.input_node = node
    self.weight = weight
    self.partial = 0

  def accum_partial(self, amount):
    """ Accumulate partial of this link. """
    if DEBUG_TRAIN:
      print("  accumulate ", self.input_node.state, amount)
    self.partial += self.input_node.state * amount

  def adjust_weight(self, multiplier):
    """ Adjust weight of this link. """
    self.weight -= multiplier * self.partial

  def zero_partial(self):
    """ Zero partial of this link. """
    self.partial = 0


class _Node(object):
  """
  A _Node in a neural network.

  Args:
    node_list (:obj:`list` of :class:`_InputLink`): The inputLinks that will
      feed into this node.

  Attributes:
    links (:obj:`list` of :class:`_InputLink`): The nodes that inputLink into
      this node.
    state (:obj:`Number`): The state of the node which, after a forward pass
      through the entire network, includes application of this node's layer's
      activation function.
      Note: For an output node this is the state before the loss function is
      applied but after any activation is applied.  In other words, for an
      output node, state holds, after a forward pass, this node's y_hat, the
      value predicted by the net that approximates the target y for the given
      example.
  """
  def __init__(self, node_list=None):
    self.links = []
    self.state = 0
    if node_list is None:
      node_list = []
    for node in node_list:
      self.links.append(_InputLink(node, 2 * random.random() - 1.0))
    if DEBUG_INST:
      print("  node created")

  def forward(self):
    """
    Feedforward for all the states of the nodes that inputLink into this node.
    """
    self.state = reduce(
        add, [link.input_node.state * link.weight for link in self.links]
    )

  def accum_partials(self, multiplier):
    """ Accumulate this node's partials. """
    if DEBUG_TRAIN:
      print("accumulating partials")
    for link in self.links:
      link.accum_partial(multiplier)

  def adjust_weights(self, mb_adjusted_learning_rate):
    """ Adjust the weights of the inputLinks incoming to this node.  """
    if DEBUG_TRAIN:
      print("adjusting weights")
    for link in self.links:
      link.adjust_weight(mb_adjusted_learning_rate)

  def zero_partials(self):
    """ Zero this node's partials. """
    for link in self.links:
      link.zero_partial()


class _Layer(object):
  """
  A layer in an instance of the Net class.

  Args:
    num_nodes (int): The number of nodes to create in this layer.
    input_layer (:obj:`_Layer`, optional): None or a layer whose nodes will
      feed into this layer's nodes.
    activation (str, optional): A string determining this layer's activation
      function.

  Attributes:
    nodes (:obj:`list` of :obj:`_Node`): The nodes of this layer.
    activation (:class:`Activate`, optional) : This layer's activation
      function or None if this is the input layer..
    aux_der: An auxillary function used when building partials while feeding
      forward.
    partials (:obj:`list` of :obj:`Number`, optional) : A list for holding the
      partial derivatives needed to update the inputsLinks to the nodes of
      the layer or None if this is the input layer.
  """
  def __init__(
      self,
      num_nodes,
      input_layer=None,
      activation=None,
      aux_der=None
  ):
    self.nodes = []
    if input_layer is None:                 # Then this is the input layer so
      for dummy_index in range(num_nodes):  #   add nodes with no inputLinks.
        self.nodes.append(_Node())
    else:                                   # This is not the input layer so
      self.activation = activation          #   we need an activation, and an
      self.aux_der = aux_der                #   and an auxillary function.
      for dummy_index in range(num_nodes):  # Add nodes to this layer and
        self.nodes.append(                  #   to each new node, connect
            _Node(input_layer.nodes)        #   every node of the input
        )                                   #   layer.

  def forward(self, xs_=None, ys_=None):
    """
    Forward the states of the nodes in the previous layer through this layer --
    and applying this nodes activation -- updating the state of each node in
    this layer.

    Args:
      xs_ (:obj:`list` of :obj:`Number`): The features of the example being fed
        though the layer.
      ys_ (:obj:`list` of :obj:`list` of :obj:`Number`): The corresponding i
        targets.
    """
    if xs_ != None:                         # Then this is the input layer
      if DEBUG_TRAIN:                       #   so just plug in the inputs.
        print("forwarding: setting inputs")
      assert len(self.nodes) == len(xs_)
      for node, x__ in zip(self.nodes, xs_):
        node.state = x__
    else:              # Then this is not the input layer so feed
      if DEBUG_TRAIN:  #   forward and apply the activation.
        print("forwarding: through non-input layers")
      for node in self.nodes:
        node.forward()
        node.state = self.activation.func(node.state)
      if ys_ != None:  # Then this is the output layer.
        assert len(ys_) == len(self.nodes)
        for idk, node in enumerate(self.nodes):
          node.accum_partials(self.aux_der(node.state, ys_[idk]))

  def backprop(self, mb_adjusted_learning_rate):
    """ For each node in this layer, adjust the incoming weights. """
    for node in self.nodes:
      node.adjust_weights(mb_adjusted_learning_rate)

  def zero_grad(self):
    """ Zero the gradient for each node of this layer. """
    for node in self.nodes:
      node.zero_partials()

class Net(object):
  """
  A fully-connected, feed-forward, neural network class.

  One recovers stochastic gradient descent using batchsize = 1; and gradient
  descent by setting batchsize equal to the number of examples in the training
  data.

  Currently supported activations:
    'None'(same as 'id'), 'ReLU', 'sigmoid', and 'tanh'.

  Currently supported loss functins:
    'MSE', 'sigmoid-MSE' (MSE stands for mean squared error).

  Args:
    nodes_per_layer (list of int) : A list of integers determining the number
      of nodes in each layer.
    activations (:obj:`lst): A list of strings one for each hidden layer fol-
      lowed by one for the output layer, each determining that layer's activ-
      ation function.
    loss (string): A string specifying the loss function to use when gauging
      accuracy of the output of model.

  Attributes:
    layers (list of :obj:`_Layer`): A list of the nets layers starting with
      the input layer, proceeding through the hidden layers. and ending with
      the output layer.
    string (str): The string to display when calling print on the model.
    loss (str): A string specifying the loss function to use when gauging the
      accuracy of the output of model.
    dloss_dphi :
  """
  def __init__(
      self,
      nodes_per_layer,
      activations,
      loss='MSE'
  ):
    self.layers = []
    self.loss = LOSS_FUNCTIONS[loss]

    assert nodes_per_layer[-1] == 1, "At most one output for now."
    assert loss in LOSS_FUNCTIONS.keys(),\
           "Invalid loss fn: must be one of " + str(LOSS_FUNCTIONS.keys())
    assert len(activations) == len(nodes_per_layer) - 1,\
           "Length of activations list should be " +str(len(nodes_per_layer)-1)\
            + "not" + str(len(activations))+"."
    assert reduce(and_, [s in ACTIVATIONS.keys() for s in activations]),\
             "No such activation: must be one of " + str(ACTIVATIONS.keys())

    if DEBUG_INST:
      print("creating an input layer with", nodes_per_layer[0], "node(s).")
    self.layers.append(_Layer(nodes_per_layer[0]))

    for i in range(1, len(nodes_per_layer)-2):
      if DEBUG_INST:
        print(
            "creating a hidden layer", i, "with", nodes_per_layer[i],
            "node(s), and activation ", str(activations[i-1]), "."
        )
      activation = ACTIVATIONS[activations[i-1]]
      self.layers.append(
          _Layer(
              nodes_per_layer[i],
              self.layers[-1],
              activation.func,
              activation.der
          )
      )

    if DEBUG_INST:
      print(
          "creating output layer with", nodes_per_layer[-1], "node(s), " +\
          "activation " + str(activations[-1]) + ", and loss " + loss + "."
      )
    activation = ACTIVATIONS[activations[-1]]
    self.layers.append(
        _Layer(
            nodes_per_layer[-1],
            self.layers[-1],
            activation,
            lambda y1, y2: self.loss.der(y1, y2) * activation.der(y1)
        )
    )

    # build a string representing the model
    self.string = "\nThe model:\n  layer 1: "+str(nodes_per_layer[0])+\
                  " input(s)\n"
    for i in range(1, len(nodes_per_layer) - 1):
      self.string += "  layer " + str(i+1)+": "+str(nodes_per_layer[i])+\
                     " nodes;  activation: "+str(activations[i-1])+"\n"
    self.string += "  layer " + str(len(nodes_per_layer)) + ": "+\
                      str(nodes_per_layer[-1])+" output node(s); "+\
                   " activation: "+str(activations[-1])+\
                   ";  loss function: "+str(loss) + "."

  def forward(self, xss, yss=None, with_grad=False):
    """
    Feed forward the xss, the features of a batch of examples. If this is called
    as part as the forward pass preceding a backpropation, then xss's correpond-
    ing targets should be passed as yss, and with_grad should be True.

    Args:
      xss (:obj:`list` of :obj:`list` of :obj:`Number`): A list of lists each of
        which is a feature from the batch being forwarded.
      yss (:obj:`list` of :obj:`list` of :obj:`Number`): The corresponding
        targets.
      with_grad (bool): If True, updates the gradients while feeding forward.

    Returns:
      y_hats (lst of lst of Numbers): The corresponding outputs.
    """
    assert len(xss[0]) == len(self.layers[0].nodes),\
        "Numer of dimensions in a feature is incorrect. Should be " +\
        str(len(self.layers[0].nodes)) + " got " + str(len(xss[0])) + "."

    y_hats = []
    for idx, xs_ in enumerate(xss):             # Feed each example xs in the
      self.layers[0].forward(xs_=xs_)           #   mini-batch into the input
      for layer in range(1, len(self.layers)):  #   layer and forward through
        if with_grad:                           #   the rest of the layers.
          self.layers[layer].forward(ys_=yss[idx])
        else:
          self.layers[layer].forward()
      y_hats.append([node.state for node in self.layers[-1].nodes])
    return y_hats

  def _forward(self, xss, yss):
    """
    Internal forward.

    Returns:
      curr_loss (Number): The average loss associated to the mini-batch with
        features xss and targets yss.
    """
    y_hats = self.forward(xss, yss, with_grad=True)
    curr_loss = 0
    for idx, ys_ in enumerate(yss):
      curr_loss += reduce(add, map(self.loss.func, y_hats[idx], ys_))
    return curr_loss/len(xss)

  def backprop(self, mb_adjusted_learning_rate):
    """ Back-propogate through layers. """
    if DEBUG_TRAIN:
      print("back propogating")
    self.layers[-1].backprop(mb_adjusted_learning_rate)

  def zero_grads(self):
    """ Zero the gradients of all layers of this net. """
    if DEBUG_TRAIN:
      print("setting gradients to zero")
    for layer in self.layers[1:]:
      layer.zero_grad()

  def learn(self, xss, yss, learning_rate=.1):
    """
    Apply one step along mini-batch gradient descent.

    Args:
      xss (:obj:`list` of :obj:`list` of :obj:`Number`): A list of lists each of
        which is a feature from the batch being forwarded.
      yss (:obj:`list` of :obj:`list` of :obj:`Number`): The corresponding
        targets.
      learning_rate (Number): Scaling factor for the gradient during descent.
    """
    assert(len(xss) == len(yss)), "Lengths of xss and yss should be the same."

    if DEBUG_TRAIN:
      print('hit learn')
    curr_loss = self._forward(xss, yss)
    self.backprop(learning_rate/len(xss))
    return curr_loss

  def __str__(self):
    return self.string


#######################  Utilility functions ##################################

#pylint: disable=too-many-arguments,too-many-locals
def train(net, xss, yss, batchsize, epochs, learning_rate, prtlns=30):
  """
  A convenient function that implements a loop for training instances of Net.
  It spews out the last prtlns current losses without the cost of
  computing the current loss when you don't really need to see it's exact value.
  It also deals with the situation when the number of samples isn't divisble by
  the mini-batch size.
  """

  n_examples = len(xss)
  thresh = epochs*n_examples/batchsize-int(prtlns*batchsize/n_examples)-1

  for i in range(int(epochs * n_examples / batchsize)):
    xyss = list(zip(xss, yss))
    random.shuffle(xyss)
    xss, yss = zip(*xyss)
    total_ave_loss = 0
    for j in range(0, n_examples, batchsize):
      xss_mb = (xss + xss[:batchsize])[j: j + batchsize]
      yss_mb = (yss + yss[:batchsize])[j: j + batchsize]
      net.zero_grads()
      loss = net.learn(xss_mb, yss_mb, learning_rate)
      total_ave_loss = (total_ave_loss + loss)/2
      if i >= thresh and j > n_examples - batchsize * prtlns:
        print('current loss: {0:12.4f}'.format(total_ave_loss))
    if i <= thresh:
      print('current loss: {0:12.4f}'.format(total_ave_loss), end='\b'*26)
  return net

###############  main  ################

def main():
  """ Run unit tests, generate some data, and test the Net class. """

  ### run the unit tests ###
  import doctest
  doctest.testmod()

  ### now generate some data and solve a linear regression ###
  num_examples = 20

  # generate some data
  xss = []
  yss = []
  stdev = 10
  for dummy_index in range(num_examples):
    x1_ = random.uniform(0, 40)
    x2_ = random.uniform(0, 60)
    xss.append([x1_, x2_])
    yss.append([2 * x1_ + 5 * x2_ + 7 + random.normalvariate(0, stdev)])

  # mean center and nomalize
  from pure_python_stats import mean_center, normalize, un_map_weights
  xmeans, xss = mean_center(xss)  # x_means is a list of the means of cols of
  xstdevs, xss = normalize(xss)   # the xss; x_stdevs holds the stand devs of
  ymeans, yss = mean_center(yss)  # of the columns; similarly for ymeans and
  ystdevs, yss = normalize(yss)   # and ystdevs.

  batchsize = 20
  net = Net([2, 1], activations=[None], loss='MSE')
  print(net)

  epochs = 100
  learning_rate = 0.1

  net = train(net, xss, yss, batchsize, epochs, learning_rate, prtlns=30)

  DEBUG_TRAIN = False

  def compute_r_squared(net, xss, yss):
    """
    Return 1-SSE/SST which is the proportion of the variance in the data
    explained by the regression hyper-plane.
    """
    ss_e = 0.0
    ss_t = 0.0

    from pure_python_stats import columnwise_means
    ymean = columnwise_means(yss)  # mean of the output variable
                                   # (which is zero if data is mean-centered)
    for idx, y__ in enumerate(yss):
      y_hat = net.forward([xss[idx]])
      ss_e = ss_e + (y__[0] - y_hat[0][0])**2
      ss_t = ss_t + (y__[0] - ymean[0])**2
    return 1.0-ss_e/ss_t

  print('1-SSE/SST =', compute_r_squared(net, xss, yss,))

  weights = [net.forward([[1, 0]])[0][0], net.forward([[0, 1]])[0][0]]
  weights = un_map_weights(weights, xmeans, xstdevs, ymeans, ystdevs)

  print('weights: ', weights[0], weights[1], weights[2],\
                                               'should be close to', 7, 2, 5)

if __name__ == '__main__':
  main()
