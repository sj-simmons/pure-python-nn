# SGD_nn.py                                                                  Simmons  Spring 18
#
# This implements a feed-forward, fully-connected neural net in pure Python that trains using
# SGD (stochastic gradient descent).

import math, random

verbose = False

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
    self.partial = 0

  def zeroPartial(self):
    self.partial = 0

  def addToPartial(self, x):
    self.partial += x

  def adjustWeight(self, learning_rate):
    self.weight = self.weight - learning_rate * self.partial
    if verbose: print("adjusting weight, partial =", self.partial)


class Node:
  """
  A Node in a neural network.

  Attributes:
    inputs (list) : A list of instances of InputLists representing all the Nodes in the neural
                    net that 'feed into' this node.
    state (number): Note: for an output node this is the state BEFORE the criterion is
                    applied.
  """

  def __init__(self, nodeList, activation = None):
    self.inputs = []
    self.state = 0
    if verbose: print("  node created")
    for node in nodeList:
      self.inputs.append(InputLink(node, 2* random.random() - 1.0))

  def setState(self, value):
    self.state = value

  def getState(self):
    return self.state

  def zeroGradient(self):
    if verbose: print("zeroing partials")
    for inputLink in self.inputs:
      inputLink.zeroPartial()

  def feedforward(self, function = None, with_grad = False, output = None ):
    """
    Feedforward for all the inputs to this instance of Node, applying the activation function
    if present.  If with_grad, then accumulate this node's gradient.

    Attributes:
      activation (string): A valid string of either an activation function (indicating that
                           this node is in a hidden layer) or a criterion (indicating that
                           this is an output node, or None (if this is an input node or a
                           hidden node with no activation.
      with_grad (boolean): Accumulate this node's gradient if True.
      output (number)    : If accumulating the gradient, we need an output.
    """
    assert output == None or with_grad, "If accumulating gradient, an output must be passed."
    if verbose: print("feeding foward. with_grad =", with_grad, "and function =", function)

    # feedforward from all the inputs to this node
    sum_ = 0
    for inputLink in self.inputs:
      sum_ += inputLink.weight * inputLink.inputNode.state
    self.setState(sum_)

    # while we are feeding forward, add contribution of this example to the partials
    if function == 'MSE':
      for inputLink in self.inputs:
        inputLink.addToPartial((sum_ - output[0]) * inputLink.inputNode.state)
    elif function == 'sigmoid':
      for inputLink in self.inputs:
        inputLink.addToPartial((sigmoid(sum_) - output[0]) * d_sigmoid(self.state) *\
                                                                   inputLink.inputNode.state)

  def adjustWeights(self, learning_rate, batchsize):
    if verbose: print("adusting weights")
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

  def __init__(self, nodes_per_layer, activations = [], batchsize = 1, criterion = 'MSE'):
    """
    A neural network class.

    One recovers stochastic gradient descent using batchsize = 1; and gradient descent by
    setting batchsize equal to the number of examples in the training data.

    Attributes
      nodes_per_layer (list):
      activations (List)    : A list of strings, currently each either 'linear' or 'sigmoid',
                              one for each hidden layer.
      batchsize (int)       : The number of examples in a batch.
      criterion (string)    : Either 'MSE' or 'sigmoid'.  TODO: add 'LogSoftMax'.
    """
    self.nodes_per_layer = nodes_per_layer
    self.inputNodes = []
    self.hiddenNodes = []
    self.outputNodes = []
    self.activations = activations
    self.criterion = criterion
    self.batchsize = batchsize

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
    """
    Apply one step along mini-batch gradient descent.

    Args:
      inputs (list): A list of lists holding the batch's inputs.
      outputs (list): A list of lists holding the batch's corresponding outputs.
      learning_rate (number): Scaling factor for the gradient during descent.
    """

    assert(len(inputs) == self.batchsize), "Number of inputs is " + str(len(inputs)) +\
                                           " but batchsize is " + str(self.batchsize)
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
      for node in self.hiddenNodes:
        node.feedforward(activation = self.activations[0], with_grad = True, output = None)
      if with_grad:
        for node in self.outputNodes:
          node.feedforward(function = self.criterion, with_grad = True, output = outputs[i])
      else:
        for node in self.outputNodes:
          node.feedforward(function = None, with_grad = False, output = None)

  def zeroGrads(self):
    if verbose: print("setting gradients to zero")
    for node in self.hiddenNodes:
      node.zeroGradient()
    for node in self.outputNodes:
      node.zeroGradient()

  def backprop(self, learning_rate):
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

    self.forward(inputs)
    total_error = 0
    for input in inputs:
      for idx in range(len(self.outputNodes)):
        total_error += (self.getOutput() - outputs[idx][0])**2
    return total_error / len(inputs)

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


if __name__ == '__main__':

  import random
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
  net = Net([1,1], batchsize = batchsize, criterion = 'MSE')

  epochs = 5000
  learning_rate = 0.05
  indices = list(range(num_examples))
  printlns = epochs*batchsize-int(30*batchsize/num_examples)-1

  for i in range(epochs * batchsize):
    random.shuffle(indices)
    if verbose: print('shuffling')
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
