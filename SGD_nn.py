# SGD_nn.py                                                   Simmons  Spring 18
#
# This implements a feed-forward, fully-connected neural net in pure Python that
# trains using SGD (stochastic gradient descent).

import math, random
from functools import reduce
from operator import add, and_

debug_inst = False   # Print debugging info during instantiation of a Net.
debug_train = False  # Print debugging info during training.


##################  Some activations and their derivatives  ####################

class Activate(object):
  """
  An activation function class.

  >>> activations = map(lambda s: Activate(s), [None, 'id', 'ReLU', 'sigmoid'])
  >>> [acitivation.func(-7) for acitivation in activations] # doctest:+ELLIPSIS
  [-7, -7, 0, 0.000911...
  >>> # The code above works but activations is a map object that is lazy evalu-
  >>> # ated; so we cannot get our hands on, for instance, the first activation,
  >>> # so that, for example,  list(activations)[0] thows an exception.
  >>> # For our application below, do this instead:
  >>> activations = [Activate(str_) for str_ in ['id', 'ReLU', 'sigmoid']]
  >>> activations[0] # doctest:+ELLIPSIS
  <__main__.Activate objec...
  >>> activations[1].func(-3)
  0
  """
  # Some activation functions, f(x):
  funcs = {
    'sigmoid': lambda x: 1 / (1 + math.exp(-x)),
    'ReLU': lambda x: max(0, x),
    'id': lambda x: x,
    None: lambda x: x,
  }

  # And their derivatives, f'(y):
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
  ders = {
    'sigmoid': lambda y: y * (1 - y),
    'ReLU': lambda y: 0 if y == 0 else 1,
    'id': lambda y: 1,
    None: lambda y: 1,
  }

  def __init__(self, activation):
    self.func = self.funcs.get(activation, '')
    self.der = self.ders.get(activation, '')


###################  Loss functions and their derivatives  #####################

class Set_loss(object):
  """
  Sets a loss function L(y_hat) = L(y_hat, y) to be used with (mini-)batches of
  examples. Also provides L'(y_hat) = dL(y_hat)/dy_hat

  Note: For a minibatch, L is one summand of the Loss function applied to the
  mini-batch.

  Args:
    y_hat (Number): The nets predicted target based on a features x.
    y (Number): The actual target of the example with feature x.
  """
  losses = {
    'MSE': lambda y_hat, y: (y_hat - y)**2,
  }

  # their derivative dJ(y_hat)/dy_hat up to a constant
  ders = {
    'MSE': lambda y_hat, y: y_hat - y
  }

  def __init__(self, loss):
    self.func = self.losses.get(loss, '')
    self.der = self.ders.get(loss, '')


##################  A few useful functions on functions  #######################

#def mult_funcs(f,g):
#  """
#  Return the product f(x)*g(x) of f(x) and g(x).
#
#  Args:
#    f(function): A function of one variable.
#    g(function): A functino of one variable.
#  >>> mult_funcs(lambda x: x**2, lambda x: 2*x)(5)
#  250
#  """
#  return lambda y1: f(y1) * g(y1)

def mult_funcs2(f,g):
  """
  Return the product f(x,y)*g(x) of f(x,y) and g(x).

  Args:
    f(function): A function of two variables
    g(function): A functino of one variable.
  >>> mult_funcs2(lambda x,y: x+y, lambda x: 3*x)(3,2)
  45
  """
  return lambda y1, y2: f(y1, y2) * g(y1)

#def mult_ders(f, g):
#   """
#   Return the product of derivative of instances of Activate and/or Set_loss
#   """
#   return mult_funcs(f.der, g.der)

#############################  The Neural Net  #################################

class _InputLink(object):
  """
  A connection from one node to another.

  Args:
    node (:obj:`Node`): The node that this inputLink is to emantate from.
    weight (:obj: `Number`): This links initial weight.

  Attributes:
    input_node (:obj:`Node`): The node that this inputLink emantates from.
    weight: (:obj:`Number`): This links weight.
    partial (:obj: `Number`): Holds the scaled partial w/r to this links
      weight.
  """
  def __init__ (self, node, weight):
    if debug_inst:
      print("    inputLink created")
    self.input_node = node
    self.weight = weight
    self.partial = 0

  def accum_partial(self, amount):
    if debug_train:
      print("  accumulate ", self.input_node.state, amount)
    self.partial += self.input_node.state * amount

  def adjust_weight(self, multiplier):
    self.weight -= multiplier * self.partial

  def zero_partial(self):
    self.partial = 0


class Node(object):
  """
  A Node in a neural network.

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
  def __init__(self, nodeList = []):
    self.links = []
    self.state = 0
    if debug_inst:
      print("  node created")
    for node in nodeList:
      self.links.append(_InputLink(node, 2 * random.random() - 1.0))

  def forward(self):
    """
    Feedforward for all the states of the nodes that inputLink into this node.
    """
    self.state = reduce(
      add, map(lambda link: link.input_node.state * link.weight, self.links)
    )

  def accum_partials(self, multiplier):
    if debug_train:
      print("accumulating partials")
    for link in self.links:
      link.accum_partial(multiplier)

  def adjust_weights(self, mb_adjusted_learning_rate):
    """ Adjust the weights of the inputLinks incoming to this node.  """
    if debug_train:
      print("adjusting weights")
    for link in self.links:
      link.adjust_weight(mb_adjusted_learning_rate)

  def zero_partials(self):
    for link in self.links:
      link.zero_partial()


class Layer(object):
  """
  A layer in an instance of the Net class.

  Args:
    num_nodes (int): The number of nodes to create in this layer.
    input_layer (:obj:`Layer`, optional): None or a layer whose nodes will
      feed into this layer's nodes.
    activation (str, optional): A string determining this layer's activation
      function.

  Attributes:
    nodes (:obj:`list` of :obj:`Node`): The nodes of this layer.
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
    input_layer = None,
    activation = None,
    aux_der = None
  ):
    self.nodes = []
    if input_layer is None:       # Then this is the input layer so add
      for i in range(num_nodes):  #   nodes that have no inputLinks.
        self.nodes.append(Node())
    else:                                      # This is not the input layer so
      self.activation = activation             #   we need an activation, and an
      self.aux_der = aux_der                   #   and an auxillary function.
      for i in range(num_nodes):         # Add nodes to this layer and
        self.nodes.append(               #   to each new node, connect
          Node(input_layer.nodes)        #   every node of the input
        )                                #   layer.

  def forward(self, xs = None, ys = None):
    """
    Forward the states of the nodes in the previous layer through this layer --
    and applying this nodes activation -- updating the state of each node in
    this layer.

    Args:
      xs (:obj:`list` of :obj:`Number`): The features of the example being fed
        though the layer.
      ys (:obj:`list` of :obj:`list` of :obj:`Number`): The corresponding i
        targets.
    """
    if xs != None:                         # Then this is the input layer
      if debug_train:
        print("forwarding: setting inputs")
      assert len(self.nodes) == len(xs)
      for node, x in zip(self.nodes, xs):  #   so just plug in the inputs.
        node.state = x
    else:                      # Then this is not the input
      if debug_train:
         print("forwarding: through non-input layers")
      for node in self.nodes:  #   layer so feed forward and
        node.forward()         #   apply the activation.
        node.state = self.activation.func(node.state)
      if ys != None:
        assert len(ys) == len(self.nodes)
        for idk, node in enumerate(self.nodes):
          node.accum_partials(self.aux_der(node.state, ys[idk]))

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
    dloss_dphi :
  """
  def __init__(
      self,
      nodes_per_layer,
      activations=[],
      batchsize=1,
      loss='MSE'
  ):
    self.layers = []
    self.batchsize = batchsize
    self.loss = Set_loss(loss)

    assert nodes_per_layer[-1] == 1, "At most one output for now."
    assert loss in Set_loss.losses.keys(),\
           "Invalid loss fn: must be one of " + str(Set_loss.losses.keys())
    assert len(activations) == len(nodes_per_layer) - 1,\
           "Length of activations list should be " +str(len(nodes_per_layer)-1)\
            + "not" + str(len(activations))+"."
    assert reduce(and_,\
             map(lambda s: s in Activate.funcs.keys(), activations)),\
             "No such activation: must be one of " + str(Activate.funcs.keys())

    if debug_inst:
      print("creating an input layer with", nodes_per_layer[0], "node(s).")
    self.layers.append(Layer(nodes_per_layer[0]))

    for i in range(1, len(nodes_per_layer)-2):
      if debug_inst:
        print(
            "creating a hidden layer", layer, "with", nodes_per_layer[i],
            "node(s), and activation ", str(activations[i-1]), "."
        )
      activation = Activate(activations[i-1])
      self.layers.append(
          Layer(
              num_per_layer[i],
              self.layers[-1],
              activation.func,
              activation.der
          )
      )

    if debug_inst:
      print(
          "creating output layer with", nodes_per_layer[-1], "node(s), " +\
          "activation " + str(activations[-1]) + ", and loss " + loss + "."
      )
    activation = Activate(activations[-1])
    self.layers.append(
        Layer(
            nodes_per_layer[-1],
            self.layers[-1],
            activation,
            mult_funcs2(self.loss.der, activation.der)
        )
    )

    # build a string representing the model
    self.string = "batchsize is "+str(batchsize)+"\nThe model:\n"+\
                  "  layer 1: "+str(nodes_per_layer[0])+" input(s)\n"
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
    for i in range(len(xss)):                  # Feed each example xs in the
      self.layers[0].forward(xs=xss[i])           #   mini-batch into the input
      for layer in range(1, len(self.layers)):  #   layer and forward through
        if with_grad:
          self.layers[layer].forward(ys=yss[i])           #   the rest of the layers.
        else:
          self.layers[layer].forward()           #   the rest of the layers.
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
    for i in range(len(xss)):
      curr_loss += reduce(add, map(self.loss.func, y_hats[i], yss[i]))
    return curr_loss/len(xss)

  def backprop(self, mb_adjusted_learning_rate):
    if debug_train:
      print("back propogating")
    self.layers[-1].backprop(mb_adjusted_learning_rate)

  def zero_grads(self):
    """ Zero the gradients of all layers of this net. """
    if debug_train:
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
    assert(len(xss) == self.batchsize), "Number of inputs is "+str(len(xss))+\
     " but batchsize is "+str(self.batchsize)
    assert(len(xss) == len(yss)), "Lengths of xss and yss should be the same."

    if debug_train:
      print('hit learn')
    curr_loss = self._forward(xss, yss)
    self.backprop(learning_rate/len(xss))
    return curr_loss

  def __str__(self):
    return self.string


#######################  Utilility functions ##################################

#pylint: disable-msg=too-many-arguments,too-many-locals
def train(net, xs, ys, batchsize, epochs, learning_rate, lines_to_print=30):
  """
  A convenient function that implements a loop for training instances of Net.
  It spews out the last lines_to_print current losses without the cost of
  computing the current loss when you don't really need to see it's exact value.
  It also deals with the situation when the number of samples isn't divisble by
  the mini-batch size.
  """

  n_examples = len(xs)
  printlns = epochs*n_examples/batchsize-int(lines_to_print*batchsize/n_examples)-1

  for i in range(int(epochs * n_examples / batchsize)):
    xys = list(zip(xs, ys))
    random.shuffle(xys)
    xs, ys = zip(*xys)
    total_ave_loss = 0
    for j in range(0, n_examples, batchsize):
      xss = (xs + xs[:batchsize])[j: j + batchsize]
      yss = (ys + ys[:batchsize])[j: j + batchsize]
      net.zero_grads()
      loss = net.learn(xss, yss, learning_rate)
      total_ave_loss = (total_ave_loss + loss)/2
      if i >= printlns and j > n_examples - batchsize * lines_to_print:
        print('current loss: {0:12.4f}'.format(total_ave_loss))
    if i <= printlns:
      print('current loss: {0:12.4f}'.format(total_ave_loss), end='\b'*26)
  return net

###############  main  ################

def main():
  ### run the unit tests ###
  import doctest
  doctest.testmod()

  ### now generate some data and solve a linear regression ###
  num_examples = 20

  # generate some data
  xs = []
  ys = []
  stdev = 10
  for i in range(num_examples): #pylint: disable=unused-variable
    x1 = random.uniform(0, 40)
    x2 = random.uniform(0, 60)
    xs.append([x1, x2])
    ys.append([2 * x1 + 5 * x2 + 7 + random.normalvariate(0, stdev)])

  # mean center and nomalize
  from Pure_Python_Stats\
         import mean_center, normalize, un_map_weights
  xmeans, xs = mean_center(xs) # x_means is a list consisting of the means of the cols of xs
  xstdevs, xs = normalize(xs) # x_stdevs holds the standard deviations of the columns
  ymeans, ys = mean_center(ys) # similarly here
  ystdevs, ys = normalize(ys) # and here

  batchsize = 20
  net = Net([2, 1], activations=[None], batchsize=batchsize, loss='MSE')
  print(net)

  epochs = 100
  learning_rate = 0.1

  net = train(net, xs, ys, batchsize, epochs, learning_rate, lines_to_print=30)

  debug_train = False  # Print debugging info during training.
  def compute_r_squared(net, xs, ys):
    """
    Return 1-SSE/SST which is the proportion of the variance in the data
    explained by the regression hyper-plane.
    """
    ss_e = 0.0
    ss_t = 0.0

    from Pure_Python_Stats import columnwise_means
    ymean = columnwise_means(ys)  # mean of the output variable
                                  # (which is zero if data is mean-centered)

    for idx, y in enumerate(ys):
      y_hat = net.forward([xs[idx]])
      ss_e = ss_e + (y[0] - y_hat[0][0])**2
      ss_t = ss_t + (y[0] - ymean[0])**2

    return 1.0-ss_e/ss_t

  print('1-SSE/SST =', compute_r_squared(net, xs, ys,))

  weights = [net.forward([[1, 0]])[0][0], net.forward([[0, 1]])[0][0]]
  weights = un_map_weights(weights, xmeans, xstdevs, ymeans, ystdevs)

  print('weights: ', weights[0], weights[1], weights[2],\
                                               'should be close to', 7, 2, 5)

if __name__ == '__main__':
  main()
