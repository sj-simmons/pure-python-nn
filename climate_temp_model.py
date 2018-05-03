# climate_temp_model.py                                                      Simmons  Spring 18

import csv
from SGD_nn import Net
from Pure_Python_Stats import mean_center, normalize, un_map_weights

# This code block reads the data from the csv file and, skipping the first line, writes the
# 2nd, 3rd, and 4th elements of each line to appropriate lists.
with open('temp_co2_data.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # Skip the first line of csvfile that holds the labels.
    xs = [] # Create empty lists to hold the inputs,
    ys = [] # and the outputs.
    for row in reader:
        xs.append([float(row[2]), float(row[3])]) # The 3rd and 4th entries are the inputs of
                                                  # each example.
        ys.append([float(row[1])]) # The 2nd entry of each line is the corresponding output.

# Note: the xs and ys are already list of lists, as we need them to be to use the Net class
#       in SDG_nn.py

# Now mean_center and normalize the data, saving the means and standard deviations of both
# input variable and the output variable.

xmeans, xs = mean_center(xs)
xstdevs, xs = normalize(xs)
ymeans, ys = mean_center(ys)
ystdevs, ys = normalize(ys)

# Add a column of ones to the inputs so that the neural net has a bias term.
xs = [[1] + x for x in xs]

# An instance of Net() which accepts 3 inputs (the 2 from the data plus 1 for the bias) and
# has one output.
net = Net([3,1])

# An unimportant helper function to sensibly print the current total error.
def printloss(loss, idx, epochs, num_last_lines = 0):
    if num_last_lines == 0: num_last_lines = epochs
    if idx < epochs - num_last_lines:
        print('current loss: {0:12f}'.format(loss), end='\b' * 26)
    else:
        print('current loss: {0:12f}'.format(loss))

epochs = 30
learning_rate = 0.1

for i in range(epochs):  # train the neural net
    for j in range(len(xs)):
        net.learn(xs[j], ys[j], learning_rate)
    printloss(net.getTotalError(xs, ys), i, epochs, 30)

# Get the weights from the trained model.
weights = net.getWeights() # Weights is list of length 3 for these data.

# Convert the weights back to those of the un-centered and un-normalized model.
weights = un_map_weights(weights, xmeans, xstdevs, ymeans, ystdevs)

# Print out the plane found by the neural net.
print("\nThe least squares regression plane found by the neural net is: "+\
        "{0:.3f} + {1:.3f}*x1 + {2:.3f}*x2".format(weights[0], weights[1], weights[2]), end='')
print(", where x1 is CO2 and x2 is SolarIrr.")

print("The actual least squares regression plane is:"+\
                                    "                  -11371.838 + 1.147*x1 + 8.047*x2.")
