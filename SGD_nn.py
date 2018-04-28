# SGD_nn.py                                                                  Simmons  Spring 18
#
# This implements a feed-forward, fully-connected neural net in pure Python.

import random

verbose = False


class InputLink:

  def __init__ (self, node, wt):

    self.inputNode = node  # node is an instance of Node
    self.weight = wt  # wt is a number


class Node:

  inputs = [] # list of InputLinks
  state = 0

  def __init__(self, nodeList):

    if verbose: print("  node created")
    for node in nodeList:
      self.inputs.append(InputLink(node, random.random()-.5))

  def setState(self, value):
    self.state = value

  def feedforward(self):

    # Feedforward from all the inputs to this Node.
    sum_ = 0
    for inputLink in self.inputs:
      sum_ += inputLink.weight * inputLink.inputNode.state
    self.state = sum_
    if verbose: print("the sum is", sum_)

  def adjustWeights(self, outputs, learning_rate):

    # compute the gradient(s)
    for idx in range(len(outputs)):
      gradient = []
      for inputLink in self.inputs:
        gradient.append(2 * (self.state - outputs[idx])\
                                             * inputLink.inputNode.state / len(self.inputs))
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

  def __init__(self, nodes_per_layer):

    assert len(nodes_per_layer) == 2 and nodes_per_layer[1] == 1,\
                                          "No hidden layers or multiple ouputs for now!"
    self.inputNodes = []
    self.hiddenNodes = []
    self.outputNodes = []

    # Populate the input nodes
    if verbose: print("populating input layer with",nodes_per_layer[0],"node(s).")
    for node in range(nodes_per_layer[0]):
      self.inputNodes.append(Node([]))

    # Populate the hidden layers
    for layer in range(1,len(nodes_per_layer)-1):
      if verbose:\
              print("populating hidden layer",layer,"with",nodes_per_layer[layer],"node(s).")
      for node in range(nodes_per_layer[layer]):
        self.outputNodes.append(Node(self.hiddenNodes))

    # Populate the ouput layer
    if verbose: print("populating output layer with",nodes_per_layer[1],"node(s).")
    for node in range(nodes_per_layer[-1]):
      if len(nodes_per_layer) < 3:  # if no hidden layers
        self.outputNodes.append(Node(self.inputNodes))
      else:
        self.outputNodes.append(Node(self.hiddenNodes))

  def learn(self, inputs, outputs, learning_rate = .01):

    self.forward(inputs)
    self.backprop(outputs, learning_rate)

  def forward(self, inputs): # generates output for the given inputs

    assert len(inputs) == len(self.inputNodes),\
        "Dimension of inputs is incorrect. Should be " + str(len(self.inputNodes)) + \
        " got " + str(len(inputs)) + "."
    for idx in range(len(inputs)): # feed in the inputs
      self.inputNodes[idx].setState(inputs[idx])
    for node in self.outputNodes:
      node.feedforward()

  def getTotalError(self):

    for idx in range(len(self.outputNodes)):
      return self.outputNodes[idx].getTotalError()

  def backprop(self, outputs, learning_rate):

    for node in self.outputNodes:
      node.adjustWeights(outputs, learning_rate)

  def getWeights(self):
    return self.outputNodes[0].getWeights()

