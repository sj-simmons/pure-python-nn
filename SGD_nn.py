# SGD_nn.py                                                                  Simmons  Spring 18
#
# This implements a feed-forward, fully-connected neural net in pure Python that trains using
# SGD (stochastic gradient descent).

import math, random
from functools import reduce

debugging = False

####################  Some activations and their derivatives  #####################

class activate:
  """ 
  An activation function class.

  >>> activations = map(lambda s: activate(s), [None, 'id', 'ReLU', 'sigmoid'])
  >>> [function.f(-7) for function in activations] # doctest:+ELLIPSIS
  [-7, -7, 0, 0.000911...
  >>> # The code above works but activations is a map object that is lazy evaluated
  >>> # so that we cannot for example get our hands on the first activation 
  >>> # so that  list(activations)[0]  thows an exception.
  >>> #
  >>> # For our application below, Do this instead:
  >>> activations = [activate(s) for s in [None, 'id', 'ReLU', 'sigmoid']]
  >>> activations[0] # doctest:+ELLIPSIS
  <__main__.activate objec...
  >>> activations[2].f(-3)
  0
  """

  # Some activation functions, f(x):
  funcs = {
    'sigmoid': lambda x: 1 / (1 + math.exp(-x)),
    'ReLU': lambda x: max(0, x),
    'tanh': lambda x: activate.funcs('sigmoid')(2*x) - activate.funcs('sigmoid')(-2*x),
    'id': lambda x: x,
    None: lambda x: x,
  }

  # Their derivatives, f'(y):
  #
  # Note: It's efficient to compute the derivative of the sigmoid when it's expressed as a 
  # function of y = sigmoid(x).  To make our code in the neural net classes nicer, we write
  # all the derivatives below as function of the output of the corresponding activation 
  # function.  So if f(x) is the activation function in question, the derivative below is
  # actually (d/dx)(f(x)) written as a function of y = f(x).  This is what we denote by f'(y).
  #
  # Now suppose that we want to take the derivative of g(f(x)) with respect to x.  Using
  # the chain rule we get:  (d/dy)(g(y)) * (d/dx)(f(x)) = g'(y) * f'(y).
  ders = {
    'sigmoid': lambda y: y * (1 - y),   
    'ReLU': lambda y: 0 if y == 0 else 1, 
    'tanh': lambda y: 1 - y**2,
    'id': lambda y: 1,
    None: lambda y: 1,
  }

  def __init__(self, activation):
  
    self.f = self.funcs.get(activation, '')
    self.df = self.ders.get(activation, '')

####################  Loss functions and their derivatives  #####################

class set_loss:

  # Some loss functions J(y, output) (for one example):
  #
  # We write these as functions of y, which is consistent with the discussion above where g(y)
  # is replaced with J(y, output). 
  losses = {
    'MSE': lambda y, output: (y - output)**2,
  }
  
  # their derivative dJ(y)/dy up to a constant
  ders = {
    'MSE': lambda y, output: y - output
  }

  def __init__(self, loss):
  
    self.f = self.losses.get(loss, '')
    self.df = self.ders.get(loss, '')

####################  The neural net  #####################

class InputLink:

  def __init__ (self, node, wt):
    self.inputNode = node  # node is an instance of Node
    self.weight = wt  # wt is a number
    self.partial = 0

  def zeroPartial(self):
    self.partial = 0

  def addToPartial(self, x):
    self.partial += x

  def adjustWeight(self, learning_rate):
    self.weight = self.weight - learning_rate * self.partial
    if debugging: print("adjusting weight, partial =", self.partial)

class Node:
  """
  A Node in a neural network.

  Attributes:
    inputs (list) : A list of instances of InputLists representing all the Nodes in the neural
                    net that 'feed into' this node.
    state (number): Note: for an output node this is the state BEFORE the loss function is
                    applied.
  """

  def __init__(self, nodeList):
    self.inputs = []
    self.state = 0
    if debugging: print("  node created")
    for node in nodeList:
      self.inputs.append(InputLink(node, 2 * random.random() - 1.0))

  def setState(self, value):
    self.state = value

  def getState(self):
    return self.state

  def zeroGradient(self):
    if debugging: print("zeroing partials")
    for inputLink in self.inputs:
      inputLink.zeroPartial()

  def feedforward(self, activation, with_grad = False, loss = None, output = None):
    """
    Feedforward for all the inputs to this instance of Node, applying the activation function
    if present.  If with_grad, then accumulate this node's gradient.

    Attributes:
      activation (function): An activation function.
      with_grad (boolean)  : Accumulate this node's gradient if True.
      loss (function)      : A function that is (single summand of a) of a loss function. 
      output (number)      : If accumulating the gradient, we need an example output.
    """
    assert output == None or with_grad, "If accumulating gradient, an output must be passed."
    assert loss == None or output != None, "Output node has no example output to compare to: "\
                                     + "loss is " + str(loss) + " but output is " + str(output)

    # feedforward from all the inputs to this node
    sum_ = 0
    for inputLink in self.inputs:
      sum_ += inputLink.weight * inputLink.inputNode.state

    y = activation.f(sum_)
    self.setState(y)

    # If loss != None then self is an output node and output is a number so we add the
    # contribution of the to the partials feeding into this node.
    if loss != None:
      for inputLink in self.inputs:
        inputLink.addToPartial(loss.df(y, output) * activation.df(y) * inputLink.inputNode.state)

    #if function == 'MSE':
    #  for inputLink in self.inputs:
    #    inputLink.addToPartial((sum_ - output[0]) * inputLink.inputNode.state)
    #elif function == 'sigmoid-MSE':
    #  for inputLink in self.inputs:
    #    s = sigmoid(sum_)
    #    inputLink.addToPartial((s - output[0]) * d_sigmoid(s) * inputLink.inputNode.state)

  def adjustWeights(self, learning_rate, batchsize):
    if debugging: print("adjusting weights")
    for inputLink in self.inputs:
      inputLink.adjustWeight(learning_rate / batchsize)

  def getWeights(self):
    weights = []
    for node in self.inputs:
      weights.append(node.weight)
    return weights

  def getTotalError(self):
    sum_squared_error = 0
    for inputLink in self.inputs:
      sum_squared_error += (self.state - inputLink.inputNode.state)**2 / len(self.inputs)
    return sum_squared_error

class Net:

  def __init__(self, nodes_per_layer, activations = [], batchsize = 1, loss = 'MSE'):
    """
    A neural network class.

    One recovers stochastic gradient descent using batchsize = 1; and gradient descent by
    setting batchsize equal to the number of examples in the training data.

    Currently supported activations: 'None'(same as 'id'), 'ReLU', and 'sigmoid',
    Currently supported loss functins: 'MSE', 'sigmoid-MSE' (MSE stands for mean squared error)

    Attributes
      nodes_per_layer (list):
      activations (List)    : A list of strings one for each hidden layer followed by one
                              for the output layer.
      batchsize (int)       : The number of examples in a batch.
      loss (string)         : A string specifying the loss function to use when gauging
                              accuracy of the output of model.
    """
    self.nodes_per_layer = nodes_per_layer
    self.inputNodes = []
    self.hiddenLayers = [[]]
    self.outputNodes = []
    self.activations = []
    self.batchsize = batchsize
    self.string = '\nThe model:\n'

    assert nodes_per_layer[-1] == 1, "At most one output for now."
    assert loss in set_loss.losses.keys() ,\
           "Invalid loss fn: must be one of " + str(set_loss.losses.keys())
    assert len(activations) == len(nodes_per_layer) - 1,\
           "Length of activations list should be " + str(len(nodes_per_layer) - 1) +\
           "not" + str(len(activations))+"."
    assert reduce(lambda x,y: x and y, map(lambda s: s in activate.funcs.keys(), activations)),\
                       "No such activation: must be one of " + str(activate.funcs.keys())

    # build a string representing the model
    self.string += "  layer 1: " + str(nodes_per_layer[0]) + " input(s)\n"
    for i in range(1, len(nodes_per_layer) - 1):
      self.string += "  layer " + str(i+1) +": " + str(nodes_per_layer[i])\
                     + " nodes;  activation: " + str(activations[i]) + "\n"
    self.string += "  layer " + str(len(nodes_per_layer)) + ": " + str(nodes_per_layer[-1])\
                   + " output node(s); " + " activation: " + str(activations[-1])\
                   + ";  loss function: " + str(loss) + ".\n"

    self.loss = set_loss(loss)
    self.activations = [activate(s) for s in activations]

    # Populate the input nodes:
    if debugging: print("populating input layer with", self.nodes_per_layer[0], "node(s).")
    for node in range(self.nodes_per_layer[0]):
      self.inputNodes.append(Node([]))

    # Populate the hidden layers:
    for layer in range(1,len(self.nodes_per_layer)-2):
      if debugging:\
        print("populating hidden layer",layer,"with", self.nodes_per_layer[layer], "node(s).")
      for node in range(self.nodes_per_layer[layer]):
        self.hiddenLayers[layer - 1].append(Node(self.inputNodes))

    # Populate the output layer:
    if debugging: print("populating output layer with", self.nodes_per_layer[1], "node(s).")
    for node in range(self.nodes_per_layer[-1]):
      if len(self.nodes_per_layer) < 3:  # if no hidden layers
        self.outputNodes.append(Node(self.inputNodes))
      else:
        self.outputNodes.append(Node(self.hiddenLayers[-1]))

  def learn(self, inputs, outputs, learning_rate = .1):
    """
    Apply one step along mini-batch gradient descent.

    Args:
      inputs (list): A list of lists holding the batch's inputs.
      outputs (list): A list of lists holding the batch's corresponding outputs.
      learning_rate (number): Scaling factor for the gradient during descent.
    """

    assert(len(inputs) == self.batchsize),\
     "Number of inputs is " + str(len(inputs)) + " but batchsize is " + str(self.batchsize)
    assert(len(inputs) == len(outputs)), "Lengths of inputs and outputs should be the same."

    self.forward(inputs, outputs, with_grad = True)
    self.backprop(learning_rate)

  def forward(self, inputs, outputs = None, with_grad = False):
    """
    Feed forward the given inputs.

    Attributes:
      inputs (list)   : A list of lists each list of which is one of this batch's examples.
      outputs (list)  : A list of lists of the corresponding outputs.
      with_grad (bool): If True, updates the gradients while feeding forward.
    """

    assert len(inputs[0]) == len(self.inputNodes),\
        "Dimension of inputs is incorrect. Should be " + str(len(self.inputNodes)) + \
                                                                " got " + str(len(inputs[0])) + "."

    for i in range(len(inputs)):
      for j in range(len(inputs[i])): # feed in the inputs
        self.inputNodes[j].setState(inputs[i][j])
      for layer in range(len(self.hiddenLayers)):  # feed forward through any hidden layers
        for node in self.hiddenLayers[layer]:
          if with_grad:
            node.feedforward(activation = self.activations[layer], with_grad = True, output = None)
          else:
            node.feedforward(activation = self.activations[layer], with_grad = False, output = None)
      for node in self.outputNodes: # feed forward through outputs
        if with_grad:
          node.feedforward(activation = self.activations[-1], with_grad = True, loss = self.loss, output = outputs[i][0])
        else:
          node.feedforward(activation = self.activations[-1])

  def zeroGrads(self):
    if debugging: print("setting gradients to zero")
    for layer in self.hiddenLayers:
      for node in layer:
        node.zeroGradient()
    for node in self.outputNodes:
      node.zeroGradient()

  def backprop(self, learning_rate):
    for layer in range(len(self.hiddenLayers)):
      for node in self.hiddenLayers[-layer]:
        if debugging: print("updating weights for hidden layer", layer)
        node.adjustWeights(learning_rate, self.batchsize)
    for node in self.outputNodes:
      node.adjustWeights(learning_rate, self.batchsize)

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
