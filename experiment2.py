# experiment2.py                                                             Simmons  Spring 18
#
# Runs experiment1 many times and then tabulates and analyzes the resulting slopes and
# intercepts.

from experiment1 import generate_data
from Pure_Python_Stats import mean_center, normalize, un_map_weights
from SGD_nn import Net
from statistics import mean, stdev

def find_least_squares_reg_line(xs, ys, epochs, learning_rate):
    """ Return the slope and intercept of the regression line. """

    # mean-center and normalize the data
    x_means, xs = mean_center(xs) # x_means is a list consisting of the means of the cols of xs
    x_stdevs, xs = normalize(xs) # x_stdevs holds the standard deviations of the columns
    y_means, ys = mean_center(ys) # similarly here
    y_stdevs, ys = normalize(ys) # and here

    # create an instance of the neural net class
    with Net([1,1], criterion = "MSE") as net:

        # Now train the neural net:
        for j in range(epochs):
            for i in range(len(xs)):
                net.zeroGrads()
                net.learn([xs[i]], [ys[i]], learning_rate)

        # The list weights below holds the intercept and slope, respectively, of the regression
        # line found by the neural net.
        weights = net.getWeights() # weights[0] is the intercept; weights[1] is the slope

    weights = un_map_weights(weights, x_means, x_stdevs, y_means, y_stdevs)
    return weights[0], weights[1]

num_samples = 5000

slopes = []
intercepts = []

for i in range(num_samples):
    print("pass: ", i+1, end='\b'*(7+len(str(i+1))))

    xs, ys = generate_data(m = 2, b = 7, stdev = 20, num_examples = 20) # generate data

    # Find the least squares reg line using stochastic gradient descent.
    b, m = find_least_squares_reg_line(xs, ys, epochs = 100, learning_rate = 0.1)

    slopes.append(m) # record slope
    intercepts.append(b) # record intercept

print("mean and st. dev. of slopes:     mean =", mean(slopes), "st. dev. =", stdev(slopes))
print("mean and st. dev. of intercepts: mean =",\
                                           mean(intercepts), "st. dev. =", stdev(intercepts))

# If you have matplotlib installed, you can uncomment next code block to see some nice graphs
# of the distribution of slopes and weights. If you don't have matplotlib installed have a
# look at experiment2.png.

#import matplotlib.pyplot as plt
#plt.subplot(1,2,1)
#plt.hist(slopes, 50, normed=1)
#plt.title("slope")
#plt.subplot(1,2,2)
#plt.hist(intercepts, 50, normed=1)
#plt.title("intercept")
#plt.show()
