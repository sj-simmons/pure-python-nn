# titanic_sigmoid_model.py                                                         Simmons 2018
#
# trains a linear model on Branton's version of Kaggle Titanic dataset
#
# this is just a linear model piped through sigmoid

import csv
from SGD_nn import Net
from Pure_Python_Stats import mean_center, normalize, un_map_weights
from random import shuffle

with open('datasets/titanic_numeric_train.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # Skip the first line of csvfile that holds the labels.
    xs = []; ys = []
    for row in reader:
        #xs.append([float(row[0])] + [float(row[i]) for i in range(2,len(row))])
        xs.append([float(row[0])] + [float(row[i]) for i in range(2,7)]) # without boat, body
        ys.append([float(row[1])])

xmeans, xs = mean_center(xs)
xstdevs, xs = normalize(xs)
#ymeans, ys = mean_center(ys)
#ystdevs, ys = normalize(ys)

#xs = [[1]+x for x in xs]  # now 9 inputs

batchsize = 10
net = Net([6,1], batchsize = batchsize, criterion = 'sigmoid-MSE')

epochs = 20
learning_rate = 0.1
num_examples = len(xs)
indices = list(range(num_examples))
printlns = epochs*batchsize-int(30*batchsize/num_examples)-1

# train
for i in range(epochs * batchsize):
    shuffle(indices); xs = [xs[idx] for idx in indices]; ys = [ys[idx] for idx in indices]
    for j in range(0, num_examples, batchsize): # about num_example/batchsize passes
        start = j % num_examples; end = start + batchsize
        in_  = (xs+xs[:batchsize])[start: end]; out  = (ys+ys[:batchsize])[start: end]
        net.zeroGrads()
        net.learn(in_, out, learning_rate)
        if i >= printlns and j > num_examples - batchsize * 30:
          loss = net.getTotalError(xs, ys)
          print('current loss: {0:12f}'.format(loss))
    if i <= printlns:
      loss = net.getTotalError(xs, ys)
      print('current loss: {0:12f}'.format(loss), end='\b' * 26)

# check accuracy on the training set
num_passengers = len(xs)
correct = 0
for i in range(num_passengers):
    net.forward([xs[i]])
    output = net.getOutput()
    if output > .5 and ys[i][0] == 1.0 or output <= .5 and ys[i][0] == 0.0:
        correct += 1
print('percentage correct on training data:', correct/num_passengers)

# read in the test set
with open('datasets/titanic_numeric_test.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # Skip the first line of csvfile that holds the labels.
    xs = []; ys = []
    for row in reader:
        #xs.append([float(row[0])] + [float(row[i]) for i in range(2,len(row))])
        xs.append([float(row[0])] + [float(row[i]) for i in range(2,7)])  # without boat, body
        ys.append([float(row[1])])

#xs = [[1]+x for x in xs]  # now 9 inputs

# check accuracy on the test set
num_passengers = len(xs)
correct = 0
for i in range(num_passengers):
    net.forward([xs[i]])
    output = net.getOutput()
    if output > .5 and ys[i][0] == 1.0 or output <= .5 and ys[i][0] == 0.0:
        correct += 1
print('percentage correct on test data:', correct/num_passengers)
