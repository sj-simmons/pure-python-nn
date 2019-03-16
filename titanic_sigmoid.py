# titanic_sigmoid.py                                                Simmons 2018
#
# trains a linear model on Branton's version of Kaggle Titanic dataset
#
# this is just a linear model piped through sigmoid

import csv
from feedforwardnn import Net, train
from liststats import mean_center, normalize

with open('datasets/titanic_numeric_train.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # Skip the first line of csvfile that holds the labels.
    xss = []; yss = []
    for row in reader:
        xss.append([float(row[0])] + [float(row[i]) for i in range(2,len(row))])
        #xs.append([float(row[0])] + [float(row[i]) for i in range(2,7)]) # without boat, body
        yss.append([float(row[1])])

xmeans, xss = mean_center(xss)
xstdevs, xss = normalize(xss)

# add bias
xss = [[1]+xs for xs in xss]  # now 9 inputs

batchsize = len(xss)
net = Net([len(xss[0]),1], activations = ['sigmoid'], loss = 'MSE')
print(net)

epochs = 200
learning_rate = 0.5

net = train(net, xss, yss, batchsize, epochs, learning_rate)

# check accuracy on the training set
correct = 0
for idx, xs in enumerate(xss):
    y_hat = net.forward([xs])[0][0]
    if y_hat > .5 and yss[idx][0] == 1.0 or y_hat <= .5 and yss[idx][0] == 0.0:
        correct += 1
print('percentage correct on training data:', correct/len(xss))

# read in the test set
with open('datasets/titanic_numeric_test.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # Skip the first line of csvfile that holds the labels.
    xss = []; yss = []
    for row in reader:
        xss.append([float(row[0])] + [float(row[i]) for i in range(2,len(row))])
        #xs.append([float(row[0])] + [float(row[i]) for i in range(2,7)])  # without boat, body
        yss.append([float(row[1])])

xss = [[1]+xs for xs in xss]  # now 9 inputs

# check accuracy on the test set
correct = 0
for idx, xs  in enumerate(xss):
    y_hat = net.forward([xs])[0][0]
    if y_hat > .5 and yss[idx][0] == 1.0 or y_hat <= .5 and yss[idx][0] == 0.0:
        correct += 1
print('percentage correct on test data:', correct/len(xss))
