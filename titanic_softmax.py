# titanic_sigmoid_model.py                                                         Simmons 2018
#
# trains a linear model on Branton's version of Kaggle Titanic dataset
#
# this is just a linear model piped through sigmoid

import csv
from ffnn import Net, train
from pure_python_stats import mean_center, normalize

with open('datasets/titanic_numeric_train.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # Skip the first line of csvfile that holds the labels.
    xs = []; ys = []
    for row in reader:
        xs.append([float(row[0])] + [float(row[i]) for i in range(2,len(row))])
        #xs.append([float(row[0])] + [float(row[i]) for i in range(2,7)]) # without boat, body
        ys.append([float(row[1])])

xmeans, xs = mean_center(xs)
xstdevs, xs = normalize(xs)
#ymeans, ys = mean_center(ys)
#ystdevs, ys = normalize(ys)

#xs = [[1]+x for x in xs]  # now 9 inputs

batchsize = 10
net = Net([len(xs[0]),1], activations = [None], loss = 'softmax')

epochs = 20
learning_rate = 0.1
num_examples = len(xs)

net = train(net, xs, ys, batchsize, epochs, learning_rate, prtlns=30)

# check accuracy on the training set
num_passengers = len(xs)
correct = 0
for i in range(num_passengers):
    y_hat = net.forward([xs[i]])[0][0]
    if y_hat > .5 and ys[i][0] == 1.0 or y_hat <= .5 and ys[i][0] == 0.0:
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
        ys.append([float(row[1])])

#xs = [[1]+x for x in xs]  # now 9 inputs

# check accuracy on the test set
num_passengers = len(xs)
correct = 0
for i in range(num_passengers):
    y_hat = net.forward([xs[i]])[0][0]
    if y_hat > .5 and ys[i][0] == 1.0 or y_hat <= .5 and ys[i][0] == 0.0:
        correct += 1
print('percentage correct on test data:', correct/num_passengers)
