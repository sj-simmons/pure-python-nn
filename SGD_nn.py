# SGD_nn.py                                                   Simmons  Spring 18
#
# This implements a feed-forward, fully-connected neural net in pure Python that
# trains using SGD (stochastic gradient descent).

import math, random
from functools import reduce
from operator import add, ,mul, and_

debugging = False
debug_inst = False # Print debugging info during instantiation of a Net.


##################  Some activations and their derivatives  ####################

class _Activate(object):
  """
  An activation function class.

  >>> activations = map(lambda s: _Activate(s), [None, 'id', 'ReLU', 'sigmoid'])
  >>> [function.f(-7) for function in activations] # doctest:+ELLIPSIS
  [-7, -7, 0, 0.000911...
  >>> # The code above works but activations is a map object that is lazy evalu
  >>> # -ated; so we cannot for example get our hands on the first activation,
  >>> # and, for example  list(activations)[0]  thows an exception.
  >>> # For our application below, do this instead:
  >>> activations = [_Activate(s) for s in [None, 'id', 'ReLU', 'sigmoid']]
  >>> activations[0] # doctest:+ELLIPSIS
  <__main__._Activate objec...
  >>> activations[2].f(-3)
  0
  """
  # Some activation functions, f(x):
  funcs = {
    'sigmoid': lambda x: 1 / (1 + math.exp(-x)),
    'ReLU': lambda x: max(0, x),
    'id': lambda x: x,
    None: lambda x: x,
  }

  # Their derivatives, f'(y):
  #
  # Note: It's efficient to compute the derivative of the sigmoid when it's ex-
  # pressed as a function of y = sigmoid(x).  To make our code in the neural net
  # classes nicer, we write all the derivatives below as function of the output
  # of the corresponding activation function.  So if f(x) is the activation
  # function in question, the derivative below is actually (d/dx)(f(x)) but
  # written as a function of y = f(x).  This is what we denote by f'(y).
  # Now suppose that we want to take the derivative of g(f(x)) with respect to
  # x.  Using the chain rule we get: (d/dy)(g(y)) * (d/dx)(f(x)) = g'(y)*f'(y)
  # where y = f(x).
  ders = {
    'sigmoid': lambda y: y * (1 - y),
    'ReLU': lambda y: 0 if y == 0 else 1,
    'id': lambda y: 1,
    None: lambda y: 1,
  }

  def __init__(self, activation):
    self.f = self.funcs.get(activation, '')
    self.df = self.ders.get(activation, '')


###################  Loss functions and their derivatives  #####################

class _SetLoss(object):
  """
  Sets a loss function L(y_hat) = L(y_hat, y) to be used with (mini-)batches of
  examples. Also provides L'(y_hat).

  We write L and L' as functions of the outputs y_hat, which is consistent with the discussion
  in the docstring of _Activate in that we write the derivative of l(y_hat, y) 
  with respect to y_hat as a function of .

  Args:
    y_hat (Number): The nets predicted target based on a features x.
    y (Number): The actual target of the example with feature x.
  """
  losses = {
    'MSE': lambda y_hat, y: (y - y_hat)**2,
  }

  # their derivative dJ(y)/dy up to a constant
  ders = {
    'MSE': lambda y_hat, y: y - y_hat
  }

  def __init__(self, loss):
    self.f = self.losses.get(loss, '')
    self.df = self.ders.get(loss, '')


#############################  The neural net  #################################

class _InputLink(object):
  """
  A connection from one node to another.

  Args:
    node (:obj:`Node`): The node that this inputLink is to emantate from.
    weight: (:obj: `Number`): This links initial weight.

  Attributes:
    inputNode (:obj:`Node`): The node that this inputLink emantates from.
    weight: (:obj:`Number`): This links weight.
  """
  def __init__ (self, node, weight):
    if debug_inst: print("     inputLink created")
    self.inputNode = node
    self.weight = weight


class Node(object):
  """
  A Node in a neural network.

  Args:
    node_list (:obj:`list` of :class:`_InputLink`): The inputLinks that will
      feed into this node.

  Attributes:
    links (:obj:`list` of :class:`_InputLink`): The nodes that inputLink into
      this node.
    state (:obj:`number`): The state of the node which, after a forward pass
      through the entire network, includes application of this node's layer's
      activation function.
      Note: For an output node this is the state before the loss function is
      applied but after any activation is applied.  In other words, for an
      output node, state holds, after a forward pass, this node's y_hat, the
      value predicted by the net that approximates the target y for the given
      example.
  """
  def __init__(self, nodeList):
    self.links = []
    self.state = 0
    if debug_inst: print("  node created")
    for node in nodeList:
      self.inputs.append(_InputLink(node, 2 * random.random() - 1.0))

  def forward(self):
    """
    Feedforward for all the states of the nodes that inputLink into this node.
    """
    self.state = reduce(add, map(lambda node: link.state*link.weight, self.links))

  def adjust_weights(self, scaled_grad):
    """
    Adjust the weights of the inputLinks incoming to this node.

    Args:
      scaled_grad (list of Numbers): the relavant gradient pre-negated and pre-
        scaled by the learning rate.
    """
    if debugging: print("adjusting weights")
    map(lambda link, val: link.weight += val, zip(self.inputs, scaled_grad))


class Layer(object):

  def __init__(self, num_nodes, input_layer, activation = None):
    """
    A layer in an instance of the Net class.

    Args:
      num_nodes (int): The number of nodes to create in this layer.
      input_layer (:obj:`Layer`, optional): A layer or None.
      activation (str, optional): A string determining this layer's activation
        function.

    Attributes:
      nodes (:obj:`list` of :obj:`Node`): The input layer for this layer.
      activation (:class:`_Activate`, optional) : This layer's activation
        function or None if this is the input layer..
      partials (:obj:`list` of :obj:`Number`, optional) : A list for holding the
        partial derivatives needed to update the inputsLinks to the nodes of
        the layer or None if this is the input layer.
    """
    self.nodes = []
    self.activation = None
    self.partials = None
    if input_layer == None:  # Then this is the input layer
      for i in xrange(num_nodes):  # so add nodes
        self.nodes.append(Node())  # that don't have any inputLinks.
    else:  # This is not the input layer
      self.partials = [0] * num_nodes  # so we will need partials
      self.activation = _Activate(activation) # and an activation.
      for i in xrange(num_nodes):  # Start adding nodes to this layer
        for node in input_layer.nodes:  # and to each new node, connect every
          self.nodes.append(Node(node)) # node of the input_layer.

  def forward(xs = None):
    """
    Forward the states of the nodes in the previous layer through this layer,
    updating the state of each node in this layer.

    Args:
      xs (:obj:`list` of :obj:`list` of :obj:`Number`): The batch's features.
   #   ys (:obj:`list` of :obj:`list` of :obj:`Number`): The batch's correspond-
   #     ing targets.
    """
    if xs == None:    # Then this is not the input layer
      for node in self.nodes: # so feed
        node.forward()        # forward
        nod.state = self.activation.f(node.state) # and apply the activation.
    else:    # This is an input layer so just plug in the inputs.
      map(lambda node, x: node.state = x, zip(self.nodes, xs))

  def backprop():
    gradient = map(

    for node in self.nodes:
      node.adjust_weights(  )



class Net(object):

  def __init__(self, nodes_per_layer, activations = [],\
                                                   batchsize = 1, loss = 'MSE'):
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
        lowed by one for the output layer determining that layer's activation
        function..
      batchsize (int): The number of examples in a batch that net will process
        during a training loop.
      loss (string): A string specifying the loss function to use when gauging
        accuracy of the output of model.

    Attributes:
      layers (list of :obj:`Layer`): A list of the nets layers starting with
        the input layer, proceeding through the hidden layers. and ending with
        the output layer.
      string (str): The string to display when calling print on the model.
      batchsize (int): The number of examples in a batch that net processes
        during a training loop.
      loss (str): A string specifying the loss function to use when gauging the
        accuracy of the output of model.
    """
    self.layers = []
    self.batchsize = batchsize
    self.string = '\nThe model:\n'
    self.loss = _SetLoss(loss)

    assert nodes_per_layer[-1] == 1, "At most one output for now."
    assert loss in _SetLoss.losses.keys() ,\
           "Invalid loss fn: must be one of " + str(_SetLoss.losses.keys())
    assert len(activations) == len(nodes_per_layer) - 1,\
           "Length of activations list should be " +str(len(nodes_per_layer)-1)\
            + "not" + str(len(activations))+"."
    assert reduce(and_,\
             map(lambda s: s in _Activate.funcs.keys(), activations)),\
             "No such activation: must be one of " + str(_Activate.funcs.keys())

    if debug_inst:
      print("creating an input layer with", nodes_per_layer[0], "nodes.")
    self.layers.append(nodes_per_layer[0]))

    for i in range(1,len(self.nodes_per_layer)-2):
      if debug_inst: print("creating a hidden layer", layer,"with",\
                    nodes_per_layer[i], "node(s).")
      self.layers.append(\
                    Layer(num_per_layer[i], self.layers[i-1], activations[i-1]))

    if debug_inst:
      print("creating output layer with", self.nodes_per_layer[-1], "node(s).")
    self.layers.append(\
                    Layer(num_per_layer[-1], self.layers[-1], activations[-1]))

    # build a string representing the model
    self.string += "  layer 1: " + str(nodes_per_layer[0]) + " input(s)\n"
    for i in range(1, len(nodes_per_layer) - 1):
      self.string += "  layer " + str(i+1) +": " + str(nodes_per_layer[i])\
                     + " nodes;  activation: " + str(activations[i-1]) + "\n"
    self.string += "  layer " + str(len(nodes_per_layer)) + ": " +\
                     str(nodes_per_layer[-1]) + " output node(s); " +\
                     " activation: " + str(activations[-1]) +\
                     ";  loss function: " + str(loss) + ".\n"

  def forward(self, xss, yss = None, with_grad = False):
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
      curr_loss (Number): The current loss assoiciated to the mini-batch with
        freatures xss and targets yss.
    """
    assert len(inputs[0]) == len(self.layers[0]),\
        "Numer of dimensions in a feature is incorrect. Should be " +\
        str(len(self.layers[0])) + " got " + str(len(inputs[0])) + "."

    if yss != None:  # Then we will compute the current loss of the mini-batch
      curr_loss = 0

    # feed each example xs in the mini-batch xss through the network and
    # accumulate this mini-batches current loss.
    for i in xrange(len(xss)):
      self.layers[0].forward(xss[i])  # feed into the input layer
      for layer in xrange(1,len(self.layers)):
        self.layers[layer].forward()   # forward through the rest of the layers
      if 'curr_loss' in locals():
        curr_loss += reduce(\
                       map(self.loss,\
                         zip(\
                           [node.state for node in self.layer[-1].nodes], yss[i]\
                         )))
    return curr_loss

  def backprop(self, learning_rate):
    self.layers[-1].backprop()


    for node in self.outputNodes:
      partials = node.adjustWeights(learning_rate, self.batchsize)
    for layer in range(len(self.hiddenLayers)):
      for node in self.hiddenLayers[-layer]:
        if debugging: print("updating weights for hidden layer", layer)
        node.adjustWeights(learning_rate, self.batchsize)

  def learn(self, xxs, yys, learning_rate = .1):
    """
    Apply one step along mini-batch gradient descent.

    Args:
      xss (:obj:`list` of :obj:`list` of :obj:`Number`): A list of lists each of
        which is a feature from the batch being forwarded.
      yss (:obj:`list` of :obj:`list` of :obj:`Number`): The corresponding
        targets.
      learning_rate (Number): Scaling factor for the gradient during descent.
    """
    assert(len(inputs) == self.batchsize),\
     "Number of inputs is " + str(len(inputs)) + " but batchsize is " + str(self.batchsize)
    assert(len(inputs) == len(outputs)), "Lengths of inputs and outputs should be the same."

    self.forward(xss, yss, with_grad = True)
    self.backprop(learning_rate)

  def zeroGrads(self):

    if debugging: print("setting gradients to zero")
    for layer in self.hiddenLayers:
      for node in layer:
        node.zeroGradient()
    for node in self.outputNodes:
      node.zeroGradient()

  def getTotalError(self, inputs, outputs):
    """
    Return the total mean squared error over of all input/outputs pairs.

    Args:
      inputs (list of lists):
      outputs (list of lists):
    """
    assert len(inputs) == len(outputs), "Length on inputs and outputs must be equal."

    total_error = 0
    for input_, output in zip(inputs, outputs) :
      self.forward([input_])
      total_error += self.loss.f(self.getOutput(), output[0])
    return total_error / len(inputs)

  def getWeights(self):
    """ Return the weights if this is a linear neural net. """

    assert len(self.nodes_per_layer) == 2,\
     "Method getWeights not implemented for networks with hidden layers. You probably don't"+\
     " really need the weights for those networks."

    return self.outputNodes[0].getWeights()

  def getOutput(self):
    return self.outputNodes[0].getState()

  def __str__(self):
    return self.string


#########################  Utilility functions ####################################

# A convenient function that implements a loop for training instances of Net.  Mainly,
# it spews out the last lines_to_print current losses without the cost of computing the
# current loss when you don't really need to see it's exact value.
def train(net, xs, ys, batchsize, epochs, learning_rate, lines_to_print = 30):

  indices = list(range(len(xs)))
  printlns = epochs*batchsize-int(30*batchsize/num_examples)-1

  for i in range(epochs * batchsize):
    random.shuffle(indices)
    if debugging: print('shuffling')
    xs = [xs[idx] for idx in indices]
    ys = [ys[idx] for idx in indices]
    for j in range(0, num_examples, batchsize): # about num_example/batchsize passes
      start = j % num_examples
      end = start + batchsize
      in_  = (xs+xs[:batchsize])[start: end]
      out  = (ys+ys[:batchsize])[start: end]
      net.zeroGrads()
      net.learn(in_, out, learning_rate)
      if i >= printlns and j > num_examples - batchsize * 30:
        loss = net.getTotalError(xs, ys)
        print('current loss: {0:12f}'.format(loss))
    if i <= printlns:
      loss = net.getTotalError(xs, ys)
      print('current loss: {0:12f}'.format(loss), end='\b' * 26)
  return net

###############  main  ################

if __name__ == '__main__':

  ### first run the unit tests ###
  import doctest
  doctest.testmod()

  ### now generate some data and solve a linear regression ###
  num_examples = 20

  # generate some data
  xs = [];  ys = []
  m = 2; b = 7; stdev = 10
  for i in range(num_examples):
    x = random.uniform(0,40)
    xs.append([x])
    ys.append([m * x + b + random.normalvariate(0,stdev)])

  # mean center and nomalize
  from Pure_Python_Stats import mean_center, normalize, un_center, un_normalize, un_map_weights
  xmeans, xs = mean_center(xs) # x_means is a list consisting of the means of the cols of xs
  xstdevs, xs = normalize(xs) # x_stdevs holds the standard deviations of the columns
  ymeans, ys = mean_center(ys) # similarly here
  ystdevs, ys = normalize(ys) # and here

  batchsize = 4
  net = Net([1,1], activations = [None], batchsize = batchsize, loss = 'MSE')
  print(net)

  epochs = 5000
  learning_rate = 0.05

  net = train(net, xs, ys, batchsize, epochs, learning_rate, lines_to_print = 30)

  def compute_r_squared(xs, ys, net):
    """
    Return 1-SSE/SST which is the proportion of the variance in the data explained by the
    regression hyper-plane.
    """
    SS_E = 0.0;  SS_T = 0.0

    from Pure_Python_Stats import columnwise_means
    ymean = columnwise_means(ys)  # mean of the output variable (which is zero if data is mean-centered)

    for i in range(len(ys)):
      net.forward([xs[i]])
      out = net.getOutput()
      SS_E = SS_E + (ys[i][0] - out )**2
      SS_T = SS_T + (ys[i][0] - ymean[0])**2

    return 1.0-SS_E/SS_T

  print('\n1-SSE/SST =', compute_r_squared(xs, ys, net))

  weights = net.getWeights()
  weights = un_map_weights(weights,xmeans, xstdevs, ymeans, ystdevs)

  print('weights:',weights[0], weights[1] )
