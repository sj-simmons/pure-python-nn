# climate_temp_model.py                                                      Simmons  Spring 18

import csv
from SGD_nn import Net
from Pure_Python_Stats import mean_center, normalize, un_map_weights

def test(test_csv_path, net):
    # This code block reads the data from the csv file and, skipping the first line, writes the
    # 1st through 14th elements of each line to appropriate lists.
    with open(test_csv_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        labels = []
        labels = next(csvfile).split(',')  # Skip the first line of csvfile that holds the labels.
        xs = []  # Create empty lists to hold the inputs,
        ys = []  # and the outputs.
        for row in reader:
            xs.append([float(row[i]) for i in range(1, 14)])  # The 2nd through 13th entries are the inputs of
            # each example.
            ys.append(float(row[0]))  # The 1st entry of each line is the corresponding output.

        # Add a column of ones to the inputs so that the neural net has a bias term.
        xs = [[1] + x for x in xs]

        for i in range(len(xs)):
            net.forward(xs[i])
            out = net.getOutput()
#            for j in range (len(out)):
            err = ys[i] - out #[j]
            print('Predicted: {1}\tActual: {0}\tError: {2}\n'.format(ys[i], out, err))

# This code block reads the data from the csv file and, skipping the first line, writes the
# 1st through 14th elements of each line to appropriate lists.
with open('ames_small.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    labels = []
    labels = next(csvfile).split(',')  # Skip the first line of csvfile that holds the labels.
    xs = [] # Create empty lists to hold the inputs,
    ys = [] # and the outputs.
    for row in reader:
        xs.append([float(row[i])for i in range(1,14)])# The 2nd through 13th entries are the inputs of
                                                  # each example.
        ys.append([float(row[0])]) # The 1st entry of each line is the corresponding output.

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

# An instance of Net() which accepts 15 inputs (the 13 from the data plus 1 for the bias) and
# has one output.
net = Net([14,1])

# An unimportant helper function to sensibly print the current total error.
def printloss(loss, idx, epochs, num_last_lines = 0):
    if num_last_lines == 0: num_last_lines = epochs
    if idx < epochs - num_last_lines:
        print('current loss: {0:12f}'.format(loss), end='\b' * 26)
    else:
        print('current loss: {0:12f}'.format(loss))

epochs = 30
learning_rate = 0.01

for i in range(epochs):  # train the neural net
    for j in range(len(xs)):
        net.learn(xs[j], ys[j], learning_rate)
    printloss(net.getTotalError(xs, ys), i, epochs)

# Get the weights from the trained model.
weights = net.getWeights() # Weights is list of length 14 for these data.

# Convert the weights back to those of the un-centered and un-normalized model.
weights = un_map_weights(weights, xmeans, xstdevs, ymeans, ystdevs)

# Print out the plane found by the neural net.
print("\nThe least squares regression plane found by the neural net is: "+\
        ' + '.join('{0:.3f}*x{1}'.format(weights[i], i) for i in range(1, len(weights))), end='\n')
print ("\nWhere:\n "+\
        ' \n '.join('x{1} is {0}'.format(labels[i], i) for i in range(1, len(weights))), end='\n')
#print(", where x1 is CO2 and x2 is SolarIrr.")

#print("The actual least squares regression plane is:"+\
#                                    "                  -11371.838 + 1.147*x1 + 8.047*x2.")

print ('Testing with ames_small.csv')
test('ames_small.csv', net)
