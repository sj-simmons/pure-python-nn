# housing_model.py                                                           Simmons  Spring 18

import csv
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

net = Net([13,1], criterion = 'MSE')

def printloss(loss, idx, epochs, num_last_lines = 0):
    if num_last_lines == 0: num_last_lines = epochs
    if idx < epochs - num_last_lines: print('current loss: {0:12f}'.format(loss), end='\b' * 26)
    else: print('current loss: {0:12f}'.format(loss))

epochs = 5000
learning_rate = 0.01

for i in range(epochs):  
    for j in range(len(xs)):
        net.learn(xs[j], ys[j], learning_rate)
    printloss(net.getTotalError(xs, ys), i, epochs, 40)


def compute_r_squared(xs, ys, net):
    """ 
    Return 1-SSE/SST which is the proportion of the variance in the data explained by the
    regression hyper-plane.
    """
    SS_E = 0.0;  SS_T = 0.0

    from Pure_Python_Stats import columnwise_means
    ymean = columnwise_means(ys)  # mean of the output variable (which is zero if data is mean-centered)

    for i in range(len(ys)):
      net.forward(xs[i])
      out = net.getOutput()
      SS_E = SS_E + (ys[i][0] - out )**2
      SS_T = SS_T + (ys[i][0] - ymean[0])**2

    return 1.0-SS_E/SS_T

print('\n1-SSE/SST =', compute_r_squared(xs, ys, net))

print()
weights = net.getWeights()
for i in range(len(weights)):
    print(weights[i])
print("len",len(weights))
print()
weights = un_map_weights(weights,xmeans, xstdevs, ymeans, ystdevs) 
for i in range(len(weights)):
    print(weights[i])
print("len",len(weights))

sum_ = 0
for i in range(13):
    net.forward([0]*i+[1]+[0]*(12-i))
    sum_ += sum_ * net.getOutput()
    print('intercept', net.getOutput())

print("here", ymeans[0] + sum_ * ystdevs[0])


# Now make a prediction
house_to_be_assessed = [2855, 0, 26690, 8, 7, 1652, 1972, 1040, 2080, 1756, 8, 841, 2]
print('\nestimated value: ${:,.2f}'.format(weights[0]+dotLists(weights[1:],house_to_be_assessed)))
