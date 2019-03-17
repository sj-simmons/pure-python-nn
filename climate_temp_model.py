# climate_temp_model.py                                       Simmons  Spring 18

import csv
from random import shuffle
from feedforwardnn import Net, train
from liststats import mean_center, normalize, un_map_weights

# This code block reads the data from the csv file and, skipping the first line,
# writes the 2nd, 3rd, and 4th elements of each line to appropriate lists.
with open('datasets/temp_co2_data.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # Skip the first line of csvfile that holds the labels.
    xss = [] # Create empty lists to hold the inputs,
    yss = [] # and the outputs.
    for row in reader:
        xss.append([float(row[2]), float(row[3])]) # The 3rd and 4th entries are
                                                   # the inputs of each example.
        yss.append([float(row[1])])  # The 2nd entry of each line is the corres-
                                     # ponding output.
# Note: the xss and yss are already list of lists, as we need them to be to use
#        the Net class feedforwardnn.py

# Now mean_center and normalize the data, saving the means and standard
# deviations of both input variable and the output variable.
xmeans, xss = mean_center(xss)
xstdevs, xss = normalize(xss)
ymeans, yss = mean_center(yss)
ystdevs, yss = normalize(yss)

# An instance of Net() which accepts 2 inputs and 1 output, and mean squared
# error for the loss.
net = Net([2,1], activations = [None], loss = 'MSE')
print(net)

train(net, xss, yss, batchsize=32, epochs=100, learning_rate=0.1)

# Get the weights from the trained model.
weights = [net.forward([[1, 0]])[0][0], net.forward([[0, 1]])[0][0]]

# Convert the weights back to those of the un-centered and un-normalized model.
weights = un_map_weights(weights, xmeans, xstdevs, ymeans, ystdevs)

# Print out the plane found by the neural net.
print("\nThe least squares regression plane found by the neural net is:\n"+\
        "     {0:.3f} + {1:.3f}*x1 + {2:.3f}*x2".\
        format(weights[0], weights[1], weights[2]), end='')
print(", where x1 is CO2 and x2 is SolarIrr.")

# To check how well the neural net did, here's the actual least-squares
# regression plane (found by solving the normal equations).
print("The actual least squares regression plane is:\n" + " " * 5 +\
      "-11371.838 + 1.147*x1 + 8.047*x2.")
