# experiment1.py                                                             Simmons  Spring 18

# This generates some data suitable for a linear regression and uses the pure Python neural net
# implementation to find the least squares regression line.

import random
from SGD_nn import Net
from Pure_Python_Stats import mean_center, normalize, un_center, un_normalize, un_map_weights

def generate_data(m = 2, b = 7, stdev = 20, num_examples = 20):
    """
    Return some 'fake' data consisting of a cloud of points scattered about the line y = mx + b
    (which is y = 2x + 7 by default).

    The x-values of the points are chosen randomly and uniformly from the interval [0,40].
    The y-values are chosen by adding to mx+b an error sampled from N(0, 20) (which is the nor-
    mal distribution with mean 0 and standard deviation stdev(=20 by default)).

    The x-values are returned as a list of lists of the form xs = [[x_1], [x_2], ... ].
    The y-values are returned as a list of lists of the form ys = [[y_1], [y_2], ... ].

    Args:
        m: The slope of the line used to generate the point cloud.
        b: The intercept of that line.
        stdev:  The standard deviation of the normal distribution used generate the errors.
        num_examples: The number of points in the cloud.

    Returns:
        xs, ys: These are both lists of lists.
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


if __name__ == '__main__':

    # Generate some data.
    xs, ys = generate_data(m = 2, b = 7, stdev = 20, num_examples = 20)

    # Mean-center and normalize the data.
    x_means, xs = mean_center(xs) # x_means is a list consisting of the means of the cols of xs
    x_stdevs, xs = normalize(xs) # x_stdevs holds the standard deviations of the columns
    y_means, ys = mean_center(ys) # similarly here
    y_stdevs, ys = normalize(ys) # and here

    # Create an instance of the neural net class with 1 input and 1 output, no hidden layer,
    # no activation function, mean squared error criterion.
    net = Net([1,1], activations = [None], batchsize = 1, loss = 'MSE')

    # An un-important helper function to sensibly print the current total error.
    def printloss(loss, idx, epochs, num_last_lines = 0):
        if num_last_lines == 0: num_last_lines = epochs
        if idx < epochs - num_last_lines:
            print('current loss: {0:12f}'.format(loss), end='\b' * 26)
        else:
            print('current loss: {0:12f}'.format(loss))

    epochs = 50
    learning_rate = 0.1

    # Now train the neural net:
    for j in range(epochs):
        for i in range(len(xs)):
            net.zeroGrads()
            net.learn([xs[i]], [ys[i]], learning_rate)
        printloss(net.getTotalError(xs, ys), j, epochs)

    # The list weights below holds the intercept and slope, respectively, of the regression
    # line found by the neural net.
    weights = net.getWeights() # weights[0] is the intercept; weights[1] is the slope

    # But since we mean_centered and normalized the data we have to scale and translate back to
    # where we started:
    # Let xx and yy denote the mean centered data with regression line determined by mm and bb.
    # Then yy = mm * xx + bb.  But xx = (x - xmeans)/xstdevs and yy = (y - ymeans)/ystdevs.
    # Substituting, we get
    #              (y - ymeans)/ystdevs = mm (x - xmeans)/xstdevs + bb.
    # After solving for y and doing some algebra, we get
    #   y = mm * ystdevs / xstdevs x - mm * xmeans * ystdevs / xstdevs + bb * ystdevs + ymeans.
    # Hence the slope and intercept we are after are:
    #               m = mm * ystdevs / xstdevs, and
    #               b = ymeans - mm* xmeans * ystdevs / xstdevs + bb * ystdevs.
    #
    # We can scale and translate the weights correctly by using un_map_weights as follows:
    # (the next line generalizes to data with more than one input variable):
    weights = un_map_weights(weights, x_means, x_stdevs, y_means, y_stdevs)

    m = weights[1]
    b = weights[0]

    # Print out the line found by the neural net.
    print("\nThe least squares regression line is: y = {0:.3f}*x + {1:.3f}\n". format(m, b))

    # Note: If you have matplotlib installed, then you can uncomment the next code block to see
    #       graphs of the scatterplot with the various lines.
    #       If you don't have matplotlib, have a look at experiment1.png.

#    import matplotlib.pyplot as plt
#    xs = [[pair[1]] for pair in xs] # Strip off the column of ones that we added above.
#    xs =  un_center(x_means, un_normalize(x_stdevs, xs)) # Move the point cloud back out to
#    ys =  un_center(y_means, un_normalize(y_stdevs, ys)) # where it was, originally.
#    xs = [x[0] for x in xs] # Convert from list of lists to just list.
#    ys = [y[0] for y in ys] # Same here.
#    plt.scatter(xs, ys)  # Display the scatterplot of the point cloud.
#    plt.plot(xs, [2 * x + 7 for x in xs], color = 'red',\
#                                       label = "The actual line 2x+7.") # Add the line y=2x+7.
#    lbl = '{0:.3f}x + {1:.3f}'.format(m, b) # make a label for the regression line.
#    # Add the regression line found by stochastic gradient descent:
#    plt.plot(xs, [m * x + b for x in xs], color = 'blue',\
#                                                     label = "The regression line "+lbl+".")
#    legend = plt.legend(loc = 'upper left', shadow = True) # Create a legend.
#    plt.show() # Show the graph on the screen.
