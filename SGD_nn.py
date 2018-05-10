# SGD_nn.py                                                                  Simmons  Spring 18
#
# This implements a feed-forward, fully-connected neural net in pure Python that trains using
# SGD (stochastic gradient descent).

import math, random

debugging = False


####################  Some activations and their derivatives  #####################

class activate:

  # Some activation functions f(x):
  func_dict = {
    'sigmoid': lambda x: 1 / (1 + math.exp(-x)),
    'ReLU': lambda x: max(0, x),
    'id': lambda x: x,
    None: lambda x: x,
  }

  # their derivatives df(x)/dx:
  der_dict = {
    'sigmoid': lambda y: y * (1 - y),   # note: this is a function of y = sigmoid(x)
    'ReLU': lambda x: 0 if x <= 0 else 1, 
    'id': lambda x: 1,
    None: lambda x: 1,
  }

  def __init__(self, activation):
  
    self.f = self.func_dict.get(activation, '')
    self.df = self.der_dict.get(activation, '')


####################  Criteria and their derivatives  #####################

class set_criterion:

  # Criteria f(x):
  crit_dict = {
    'MSE': lambda x, output: (x - output)**2,
  }
  
  # their deivative df(x)/dx up to a constant
  der_dict = {
    'MSE': lambda x, output: x - output
  }

  # df(g(x))/dx * g'(x)

  def __init__(self, criterion):
  
    self.f = self.crit_dict.get(criterion, '')
    self.df = self.der_dict.get(criterion, '')


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
    state (number): Note: for an output node this is the state BEFORE the criterion is
                    applied.
  """

  def __init__(self, nodeList, activation = None):
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

  def feedforward(self, activation = None, with_grad = False, criterion = None, output = None):
    """
    Feedforward for all the inputs to this instance of Node, applying the activation function
    if present.  If with_grad, then accumulate this node's gradient.

    Attributes:
      activation (function): An activation function.
      with_grad (boolean)  : Accumulate this node's gradient if True.
      criterion (function) : A function that is (single summand of a) of a criterion. 
      output (number)      : If accumulating the gradient, we need an output.
    """
    assert output == None or with_grad, "If accumulating gradient, an output must be passed."
    if debugging: print("feeding foward. with_grad =", with_grad, "and function =", function)

    # feedforward from all the inputs to this node
    sum_ = 0
    for inputLink in self.inputs:
      sum_ += inputLink.weight * inputLink.inputNode.state

    self.setState(activation(sum_))

    # If criterion != None then output is a number and self is an output node so we add the
    # contibution of the to the partials feeding into this node.
    if criterion != None:
      for inputLink in self.inputs:
        inputLink.addToPartial(criterion.df(sum_, output) * inputLink.inputNode.state)

    #if function == 'MSE':
    #  for inputLink in self.inputs:
    #    inputLink.addToPartial((sum_ - output[0]) * inputLink.inputNode.state)
    #elif function == 'sigmoid-MSE':
    #  for inputLink in self.inputs:
    #    s = sigmoid(sum_)
    #    inputLink.addToPartial((s - output[0]) * d_sigmoid(s) * inputLink.inputNode.state)

  def adjustWeights(self, learning_rate, batchsize):
    if debugging: print("adusting weights")
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
      criterion (string)    : Either 'MSE' or 'sigmoid-MSE'.  TODO: add 'LogSoftMax'.
    """
    self.nodes_per_layer = nodes_per_layer
    self.inputNodes = []
    self.hiddenLayers = [[]]
    self.outputNodes = []
    self.activations = []
    self.batchsize = batchsize

    assert len(nodes_per_layer) <= 3, "At most 3 layers for now."
    assert nodes_per_layer[-1] == 1, "At most one output for now."

    assert criterion in set(['MSE', 'MSE', 'softMax']),\
        "Currently, the criterion must be 'MSE or 'sigmoid-MSE'."
    assert criterion != 'softMax', "softMax not yet implemented."
    self.criterion = set_criterion(criterion)

    if len(nodes_per_layer) > 2:  # if there are hidden layers
      assert activations[0] in set([None, 'sigmoid', 'ReLU']), "No such activation."
      assert len(activations) == len(nodes_per_layer) - 2,\
        "Length of activations list should be " + str(len(nodes_per_layer) - 2) + "."
      self.activations = map(activate, activations)

    # Populate the input nodes
    if debugging: print("populating input layer with", self.nodes_per_layer[0], "node(s).")
    for node in range(self.nodes_per_layer[0]):
      self.inputNodes.append(Node([]))

    # Populate the hidden layers
    for layer in range(1,len(self.nodes_per_layer)-1):
      if debugging:\
        print("populating hidden layer",layer,"with", self.nodes_per_layer[layer], "node(s).")
      for node in range(self.nodes_per_layer[layer]):
        self.hiddenLayers[layer - 1].append(Node(self.inputNodes, activation = activations[layer - 1]))

    # Populate the ouput layer
    if debugging: print("populating output layer with", self.nodes_per_layer[1], "node(s).")
    for node in range(self.nodes_per_layer[-1]):
      if len(self.nodes_per_layer) < 3:  # if no hidden layers
        self.outputNodes.append(Node(self.inputNodes, activation = None))
      else:
        self.outputNodes.append(Node(self.hiddenLayers[-1], activations[-1]))

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
      for layer in range(len(self.hiddenLayers)):  # feed forward through any hidden layers
        for node in self.hiddenLayers[layer]:
          if with_grad:
            node.feedforward(activation = self.activations[layer], with_grad = True, output = None)
          else:
            node.feedforward(activation = None, with_grad = False, output = None)
      for node in self.outputNodes: # feed forward through outputs
        if with_grad:
          node.feedforward(activation = None, with_grad = True, criterion = self.criterion, output = outputs[i][0])
        else:
          node.feedforward(activation = None, with_grad = False, criterion = self.criterion, output = None)

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
