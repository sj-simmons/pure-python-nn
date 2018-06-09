# titanic_sigmoid_model.py                                      Simmons May 2018

import csv
from nnff import Net
from llstats import mean_center, normalize
from operator import indexOf
from nnutils import train

with open('datasets/titanic_numeric_train.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # Skip the first line of csvfile that holds the labels.
    xs = []; ys = []
    for row in reader:
        xs.append([float(row[0])] + [float(row[i]) for i in range(2,len(row))])
        #xs.append([float(row[0])] + [float(row[i]) for i in range(2,7)]) # without boat, body
        if row[1] == 1:
          ys.append([1,0])
        else:
          ys.append([0,1])

xmeans, xs = mean_center(xs)
xstdevs, xs = normalize(xs)
#ymeans, ys = mean_center(ys)
#ystdevs, ys = normalize(ys)

#xs = [[1]+x for x in xs]  # now 9 inputs

batchsize = 2
net = Net([len(xs[0]),2], activations = ['softmax'], loss = 'MSE')

epochs = 1
learning_rate = 0.1
num_examples = len(xs)

net = train(net, xs, ys, batchsize, epochs, learning_rate, prtlns=30)

# check accuracy on the training set
num_passengers = len(xs)
correct = 0
for i in range(num_passengers):
    y_hat = net.forward([xs[i]])[0]
    if indexOf(y_hat, max(y_hat)) == indexOf(y,max(y)):
        correct += 1
print('percentage correct on training data:', correct/num_passengers)

# read in the test set
with open('datasets/titanic_numeric_test.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # Skip the first line of csvfile that holds the labels.
    xs = []; ys = []
    for row in reader:
        xs.append([float(row[0])] + [float(row[i]) for i in range(2,len(row))])
        #xs.append([float(row[0])] + [float(row[i]) for i in range(2,7)])  # without boat, body
        if row[1] == 1:
          ys.append([1,0])
        else:
          ys.append([0,1])

#xs = [[1]+x for x in xs]  # now 9 inputs

# check accuracy on the test set
num_passengers = len(xs)
correct = 0
for i in range(num_passengers):
    y_hat = net.forward([xs[i]])[0]
    if indexOf(y_hat, max(y_hat)) == indexOf(y,max(y)):
        correct += 1
print('percentage correct on test data:', correct/num_passengers)
