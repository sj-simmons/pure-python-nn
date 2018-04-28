#experiment1.py                                                              Simmons  Spring 18

# This generates some data suitable for a linear regression and uses the pure Python neural net
# implementation to find the least squares regression line.

from SGD_nn import Net
from Pure_Python_Stats import mean_center, normalize, un_normalize
import random

def generate_data(m = 2, b = 7, stdev = 20, num_examples = 20):
    """
    Return some 'fake' data consisting of a cloud of points scattered about the line y = mx + b
    (which is y = 2x + 7 by default).

    The x-values of the points are chosen randomly and uniformly from the interval [0,40].
    The y-values are chosen by adding to mx+b a error sampled from N(0, 20) (which is the nor-
    mal distribution with mean 0 and standard deviation stdev(=20 by default).

    The x-values are returned as a list of lists of the form xs = [[x_1], [x_2], ... ].
    The y-values are returned as a list of lists of the form ys = [[y_1], [y_2], ... ].

    Args:
    m - the slope of the line used to generate the point cloud.
    b - the intercept of that line.
    stdev - the standard deviation of the normal distribution used generate the errors.
    num_examples - the number of points in the cloud.

    Returns:
    xs, ys - these are both lists of lists.
    """
    xs = []  # this list will hold the x-values
    ys = []  # this one will hold the y-values

    for i in range(num_examples):

        # Append an x-value drawn uniformly from the interval [0,40].
        x = random.uniform(0,40)
        xs.append([x])

        # Append the corresponding point on the line plus some error.
        ys.append([m * x + b + random.normalvariate(0,stdev)])

    return xs, ys

xs, ys = generate_data(stdev = 1)

# mean-center and normalize the data
x_means, xs = mean_center(xs) # x_means is a list consisting of the means of the columns of xs.
x_stdevs, xs = normalize(xs) # x_stdevs holds the standard deviations of the columns.
y_means, ys = mean_center(ys) # similarly here
y_stdevs, ys = normalize(ys) # and here

# The xs and ys should now be mean centered and normalized. You can check that by uncommenting
# the next five lines.
#new_x_means, _ = mean_center(xs); print("new x means and std devs: ", new_x_means, end=' ')
#new_x_stdevs, _ = normalize(xs); print(new_x_stdevs)
#new_y_means, _ = mean_center(ys); print("new y means and std devs: ", new_y_means, end=' ')
#new_y_stdevs, _ = normalize(ys); print(new_y_stdevs)
#exit()

# Append a column of 1s to xs, which computes the bias when training.
xs = [[1, x[0]] for x in xs]

# create an instance of the neural net class
net = Net([2,1])

epochs = 10

# An unimportant function to print the current total error sensibly
def printloss(loss):
    if j < epochs - 10:
        print('current loss: {0:12f}'.format(loss), end='\b' * 26)
    elif j == epochs - 10:
        print('\ncurrent loss: {0:12f}'.format(loss))
    else:
        print('              {0:12f}'.format(loss))

# Now train the neural net:
for j in range(epochs):
    for i in range(len(xs)):
        net.learn(xs[i], ys[i], .1) # the last argument is the learning rate
    printloss(net.getTotalError())

# The list weights below holds the intercept and slope, respectively, of the regression line
# found by the neural net.
weights = net.getWeights() # weights[0] is the intercept; weights[1] is the slope

# But since we mean_centered and normalized the data we have to scale and translate back to
# where we started:
# Let xx and yy denote the mean centered data with regression line determined by mm and bb.
# Then yy = mm * xx + bb.  But xx = (x - xmeans)/xstdevs and yy = (y - ymeans)/ystdevs.
# Substituting, we get
#              (y - ymeans)/ystdevs = mm (x - xmeans)/xstdevs + bb.
# After solving for y and doing some algebra, we get
#    y = mm * ystdevs / xstdevs x - mm * xmeans * ystdevs / xstdevs + bb * ystdevs + ymeans.
# Hence the slope and intercept we are after are:
#               m = mm * ystdevs / xstdevs, and
#               b = ymeans - mm* xmeans * ystdevs / xstdevs + bb * ystdevs.

b = y_means[0] - weights[1] * x_means[0] * y_stdevs[0] / x_stdevs[0] + weights[0] * y_stdevs[0]
# We can scale and translate weights[1:] correctly by using un_normalize as follows:
# (the next line generalizes to data with more than one input variable):
m = un_normalize(weights[1:], x_stdevs, y_stdevs)

# Print out the line found by the neural net.
print("\nThe least squares regression line is: y = {0:.3f}*x + {1:.3f}\n".format(m[0], b))
