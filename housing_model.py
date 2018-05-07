# housing_model.py                                                           Simmons  Spring 18

import csv
from random import shuffle
from SGD_nn import Net
from Pure_Python_Stats import mean_center, normalize, un_map_weights, dotLists

with open('AmesHousing.csv') as csvfile:
#with open('ames_small.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # Skip the first line of csvfile that holds the labels.
    xs = [] # Create empty lists to hold the inputs,
    ys = [] # and the outputs.
    for row in reader:
        ys.append([float(row[0])]) # The 1st entry of each line is the corresponding output.
        xs.append([float(row[i]) for i in range(1,len(row))]) # The rest are inputs. 

xmeans, xs = mean_center(xs)
xstdevs, xs = normalize(xs)
ymeans, ys = mean_center(ys)
ystdevs, ys = normalize(ys)

batchsize = 200
net = Net([13,1], batchsize = batchsize, criterion = 'MSE')

epochs = 10
learning_rate = 0.001
num_examples = len(xs)
indices = list(range(num_examples))
printlns = epochs*batchsize-int(30*batchsize/num_examples)-1
print(printlns, epochs * batchsize)

for i in range(epochs * batchsize):
    shuffle(indices)
    xs = [xs[idx] for idx in indices]
    ys = [ys[idx] for idx in indices]
    for j in range(0, num_examples, batchsize): # about num_examples/batchsize passes
        start = j % num_examples
        end = start + batchsize
        in_  = (xs+xs[:batchsize])[start: end]
        out  = (ys+ys[:batchsize])[start: end]
        net.zeroGrads()
        net.learn(in_, out, learning_rate)
        if i >= printlns and j > num_examples - batchsize * 30:
          loss = net.getTotalError(xs, ys)
          print('current loss: {0:12f}'.format(loss))
    if i < printlns:
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

# Now make a prediction
house_to_be_assessed = [2855, 0, 26690, 8, 7, 1652, 1972, 1040, 2080, 1756, 8, 841, 2]
print('\nestimated value: ${:,.2f}'.format(weights[0]+dotLists(weights[1:],house_to_be_assessed)))
