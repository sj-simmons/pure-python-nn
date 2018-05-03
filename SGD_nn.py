# SGD_nn.py                                                                  Simmons  Spring 18
#
# This implements a feed-forward, fully-connected neural net in pure Python that trains using
# SGD (stochastic gradient descent).
#
# (4/29/18) does not support hidden layers.
# (4/30/18) now supports a hidden layer but no non-linearity yet; so there currently no benefit
#           or point in using a hidden layer.
# (5/2/18) Should now work with TWO-layer networks with ONE output node with criterion either
#          MSE (Mean Squared Error) of sigmoid (poor's mans softmax).
#          TODO:  Allow a single hidden layer with either sigmoid or linear activations.
#          TODO:  Allow mulitple output with a logsoftmax criterion.

import math, random

verbose = False

# Some functions:

def sigmoid(x):
  """ Return the value of the sigmoid function evaluated at the input x. """
  return 1 / (1 + math.exp(-x))

def d_sigmoid(y):
  """
  Return the deriviative of the sigmoid function as it depends on the y-value (not the x-value)
  of a point on the graph of the sigmoid function: y = sigmoid(x).
  """
  return y * (1 - y)

def ReLU(x):
  """ Rectified Linear Unit """
  return max(0, x)


class InputLink:

  def __init__ (self, node, wt):

    self.inputNode = node  # node is an instance of Node
    self.weight = wt  # wt is a number


class Node:
  """
    A Node in a neural network.

    Attributes:
      inputs: a list of instances of InputLists representing all the Nodes in the neural net
              the 'feed into' this node.
      state: a number.  Note: for an output node this is the state BEFORE the criterion is
             applied.
  """

  def __init__(self, nodeList, activation = None):

    self.inputs = []
    self.nodeList = nodeList
    self.state = 0
    if verbose: print("  node created")
    for node in nodeList:
      self.inputs.append(InputLink(node, random.random() - 0.5))

  def setState(self, value):
    self.state = value

  def getState(self):
    return self.state

  def feedforward(self, activation = None):

    # Feedforward from all the inputs to this Node.
    sum_ = 0
    for inputLink in self.inputs:
      sum_ += inputLink.weight * inputLink.inputNode.state
    self.setState(sum_)
    if verbose: print("the sum is", sum_)

  def adjustWeights(self, outputs, learning_rate, criterion = 'MSE'):

    # compute the gradient(s)
    for idx in range(len(outputs)):
      gradient = []
      for inputLink in self.inputs:
        if criterion == 'MSE':
          gradient.append((self.state - outputs[idx]) * inputLink.inputNode.state)
        elif criterion == 'sigmoid':
          gradient.append((sigmoid(self.state) - outputs[idx]) * d_sigmoid(self.state) *\
                                                                     inputLink.inputNode.state)
    if verbose: print("the gradient is", gradient)

    # update weights
    for idx in range(len(self.inputs)):
      self.inputs[idx].weight -= learning_rate * gradient[idx]

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

  def __init__(self, nodes_per_layer, activations = [], criterion = 'MSE'):
    """
    A neural network class.

    Attributes
      nodes_per_layer (list)
      activations (List): A list of strings, currently each either 'linear' or 'sigmoid',
                          one for each hidden layer.
      criterion (string): Either 'MSE' or 'sigmoid'.  TODO: add 'LogSoftMax'.
    """
    self.nodes_per_layer = nodes_per_layer
    self.inputNodes = []
    self.hiddenNodes = []
    self.outputNodes = []
    self.activations = activations
    self.criterion = criterion

    assert len(nodes_per_layer) <= 3, "At most 3 layers for now."
    assert nodes_per_layer[-1] == 1, "At most one output for now."
    assert criterion in set(['MSE', 'sigmoid']),\
        "Currently, the criterion must be 'MSE or 'sigmoid'."
    if len(nodes_per_layer) > 2:
      assert activations[0] in set([None, 'sigmoid']), "No such activation."
      assert len(activations) == len(nodes_per_layer) - 2,\
        "Length of activations list should be " + str(len(nodes_per_layer) - 2) + "."

    # Populate the input nodes
    if verbose: print("populating input layer with", self.nodes_per_layer[0], "node(s).")
    for node in range(self.nodes_per_layer[0]):
      self.inputNodes.append(Node([]))

    # Populate the hidden layers
    for layer in range(1,len(self.nodes_per_layer)-2):
      if verbose:\
        print("populating hidden layer",layer,"with", self.nodes_per_layer[layer], "node(s).")
      for node in range(self.nodes_per_layer[layer]):
        self.hiddenNodes.append(Node(self.inputNodes, activation = activations[layer]))

    # Populate the ouput layer
    if verbose: print("populating output layer with", self.nodes_per_layer[1], "node(s).")
    for node in range(self.nodes_per_layer[-1]):
      if len(self.nodes_per_layer) < 3:  # if no hidden layers
        self.outputNodes.append(Node(self.inputNodes, activation = None))
      else:
        self.outputNodes.append(Node(self.hiddenNodes, activations[-1]))

  def learn(self, inputs, outputs, learning_rate = .1):

    self.forward(inputs)
    self.backprop(outputs, learning_rate)

  def forward(self, inputs): # generates output for the given inputs

    assert len(inputs) == len(self.inputNodes),\
        "Dimension of inputs is incorrect. Should be " + str(len(self.inputNodes)) + \
        " got " + str(len(inputs)) + "."
    for idx in range(len(inputs)): # feed in the inputs
      self.inputNodes[idx].setState(inputs[idx])
    for node in self.hiddenNodes:
      node.feedforward(activation = self.activations[0])
    for node in self.outputNodes:
      node.feedforward()

  def getTotalError(self, inputs, outputs):
    """
    Return the total mean squared error over of all input/outputs pairs.

    Args:
      inputs (list of lists):
      outputs (list of lists):
    """
    assert len(inputs) == len(outputs), "Length on inputs and outputs must be equal."

    total_error = 0
    for input in inputs:
      self.forward(input)
      for idx in range(len(self.outputNodes)):
        total_error += (self.getOutput() - outputs[idx][0])**2
    return total_error / len(inputs)

  def backprop(self, outputs, learning_rate):

    for node in self.outputNodes:
      node.adjustWeights(outputs, learning_rate, self.criterion)

  def getWeights(self):

    assert len(self.nodes_per_layer) == 2,\
     "Method getWeights not implemented for networks with hidden layers. You probably don't"+\
     " really need the weights for those networks."
    return self.outputNodes[0].getWeights()

  def getOutput(self):

    output = self.outputNodes[0].getState()
    if self.criterion == 'sigmoid':
      output = sigmoid(output)
    return output

  # The following two methods allow one to create instances of this class within a
  # Python 'with' statement.
  # Using a 'with' statement is not necessary if you just want to instantiate an
  # instance of Net() and train it only once, as in experiment1.py.
  # But if you want to instantiate an instance and re-use in multiple times, as in
  # experiment1.py, then, technically, one does well to clean things up in-between
  # re-uses by employing a 'with' statement. (See experiment2.py).
  # This has to do with the way garbage collection works in Python: specifically,
  # objects are deleted not when the go our of scope but when all references to
  # them have been removed.

  def __enter__(self):

    return self

  def __exit__(self, *args):

    self.inputNodes = []
    self.hiddenNodes = []
    self.outputNodes = []
    activations = []
    criterion = ''
