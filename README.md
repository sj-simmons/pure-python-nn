# pure-python-nn
A fully-connected, feed-forward neural net implementation in pure Python.

Written with Branton and his Spring 18 AI class.

Notes:
  * All code in this repo runs in Python 3.
  * You can pull this entire repo down to your computer using git by typing at the command
    line:
    ``` bash
    git clone http://github.com/sj-simmons/pure-python-nn
    ```
    Then enter `cd pure-python-nn` and run programs with, for example, `python3 experiment1.py`
    (assuming that `python3` is in your `PATH`; otherwise you can run and/or modify the code in IDLE).

Code summary:
  * [ffnn.py](ffnn.py): The building blocks of a feed-forward neural net that trains via stochastic gradient descent.
    Along with some functions to create non-linearity, the following classes are defined:
    * `Net` -- The neural net, which is sequence of instances of the `Layer` class.
    * `Layer` -- consisting of instances of the `Node` class.
    * `Node` -- essentially a list of instances of `InputLink`.
    * `InputLink` -- a small class with attributes `inputNode`, `weight`, and `partial`, instances of which connect the
      nodes from one layer to the next layer in a net.

    Notes:
    * The inputs and outputs to an instance of `Net` are assumed to be lists of list, one list for
      each example in the data set.
    * The `Net` class trains using mini-batch gradient descent.  Of course, you can recover stochastic gradient descent
      with batchsize = 1; or, gradient descent, by setting batchsize to the number of examples in your data.
    * Examples of instantiating the `Net` class:
      * `net = Net([8,1], batchsize = 1, criterion = 'MSE')` -- A model with 8 inputs, 1 ouput, no hidden layers, mean-squared-error criterion
        that trains using stochastic gradient descent.
      * `net = Net([15,1], batchsize = 20, criterion = 'sigmoid-MSE')` -- A model with 15 inputs, 1 output, no hidden layers that trains with
        mini-batch gradient descent using mean-squared-error criterion but first applying the sigmoid function to the output.  This is a
        [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) model
        (see also [here](https://www.encyclopediaofmath.org/index.php/Logistic_regression)) and is often referred to as a
        [single-layered perceptron](https://en.wikipedia.org/wiki/Logistic_regression#As_a_single-layer_perceptron) network.
    * One trains an instance of the `Net` class using the `learn` method:
      ``` python
      for ...
          ...
          net.learn(inputs_, outputs, learning_rate)
      ```
      where `inputs_` and `outputs` are batches of examples -- so lists of lists (of examples of the data).
    * In your training loop, you must zero out the gradients before learning:
      ``` python
      for ...
          ...
          net.zeroGrads()
          net.learn(xs, ys, learning_rate)
      ```

    TODO:
    * ~~Implement mini-batch gradient descent~~.
    * Implement learning rate decay.
      * It might be best to implement this exterior to `ffnn.py`.  The `learn` method in `ffnn.py` accepts the learning
        rate as a parameter that can be changed on the fly (while training).
    * Implement a single hidden layer.
    * Implement multiple outputs and softmax/logsoftmax.
    * Implement a multiple hidden layers.
    * Implement bias nodes in 2nd, 3rd, ... hidden layers.
  * [llstats.py](llstats.py): This is small library of functions, written in pure Python,
    that are useful, for example, for mean-centering and normalizing data in the form of lists of lists.

Neural net examples:
  * [climate_temp_model.py](climate_temp_model.py) -- A global average climate temperature model trained on these data:
    [temp_co2_data.csv](datasets/temp_co2_data.csv).
  * [housing_model.py](housing_model.py) -- A housing valuation model based on the Ames, Iowa housing data set:
    [AmesHousing.csv](datasets/AmesHousing.csv).
  * [titanic_model.py](titanic_model.py) -- Predicts survival after training on numeric versions of Branton's
    Titanic datasets.

Data sets:
  * [titanic_train.csv](datasets/titanic_train.csv) -- Branton's original Titanic training set.
  * [titanic_test.csv](datasets/titanic_test.csv) -- Branton's original Titanic test set.
  * [titanic_numeric_train.csv](datasets/titanic_numeric_train.csv) -- same as Branton's original training set but with numeric fields.
  * [titanic_numeric_test.csv](datasets/titanic_numeric_test.csv) -- same as Branton's original test set but with numeric fields.
  * [titanic_to_numeric.py](datasets/titanic_to_numeric.py) -- this was used to generate the *numeric* versions. (Uses `pandas`.)

Experiments:
  * [experiment1.py](experiment1.py) -- generates fake data appropriate for linear regression on uses an instance of the
    `Net` class in [ffnn.py](ffnn.py) to find the least-squares regression line for the data.
  * [experiment2.py](experiment2.py) -- runs experiment1 many times and tabulates and analyses the mean and variation
    of the slopes and standard deviations of the resulting regression lines.
