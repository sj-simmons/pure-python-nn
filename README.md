# pure-python-nn
A simple fully-connected, feed-forward neural net implementation in pure Python.

Written with Branton and his Spring 18 AI class.

Notes:
  * All code in this repo runs in Python 3, and not necessarily in Python 2.
  * You can conveniently pull this entire repo down to your computer using git by typing at the command
    line:
    ```
    git clone http://github.com/sj-simmons/pure-python-nn
    ```
    Then enter `cd pure-python-nn` and run programs with, for example, `python3 experiment1.py`
    (assuming that `python3` is in your `PATH`; otherwise you should be able to open, run,
    and edit the code in IDLE).

Code summary:
  * [SGD_nn.py](SGD_nn.py): The building blocks of a feed forward neural net that trains via SGD (stochastic gradient descent).
    Along with some functions to create non-linearity, the following classes are defined:
    * `Net` -- The neural net, which is built from layers (lists) of instances of the `Node` class.
    * `Node` -- essentially a list of instances of `InputLinks` along with some methods.
    * `InputLink` -- a small class with attributes `weight` and `inputNode`, instances of which connect the
      instances of Node that make up the `Net`.

    Notes:
    * The inputs and outputs to an instance of Net are assumed to be lists of list, one list for
      each example in the data set.
    * The `Net` class trains using mini-batch gradient descent.  Of course, you can recover SGD (stochastic gradient descent)
      with batchsize = 1; or, gradient descent, by setting batchsize to the number of examples in your data.
    * In your training loop, you must zero our the gradients before learning:
      ```
      net.zeroGrads()
      net.learn(xs, ys, learning_rate)
      ```
      where `net` is your instance of the `Net` class.

    TODO:
    * ~~Implement mini-batch gradient descent.~~
    * Implement learning rate decay.  
      * It might be best to implement this exterior to `SGD.py`.  The `learn` method in `SGD.py` accepts the learning
        rate as a paramater that can be changed on the fly (while training).
    * Implement a single hidden layer.
    * Implement multiple outputs and logSoftMax.
    * Implement a multiple hidden layers.
  * [Pure_Python_Stats.py](Pure_Python_Stats.py): This is small library of functions, written in pure Python,
    that are useful, for example, for mean-centering and normalizing data in the form of lists of lists.

Data sets:
  * [titanic_train.csv](datasets/titanic_train.csv) -- Branton's original Titanic training set.
  * [titanic_test.csv](datasets/titanic_test.csv) -- Branton's original Titanic test set.
  * [titanic_numeric_train.csv](datasets/titanic_numeric_train.csv) -- same as Branton's original training set but with numeric fields.
  * [titanic_numeric_test.csv](datasets/titanic_numeric_test.csv) -- same as Branton's original test set but with numeric fields.
  * [titanic_to_numeric.py](datasets/titanic_to_numeric.py) -- this was used to generate the *numeric* versions. (Uses `pandas`.)

Neural net examples:
  * [climate_temp_model.py](climate_temp_model.py) -- A global average climate temperature model trained on these data:
    [temp_co2_data.csv](datasets/temp_co2_data.csv).
  * [housing_model.py](housing_model.py) -- A housing valuation model based on the Ames, Iowa housing data set:
    [AmesHousing.csv](datasets/AmesHousing.csv).

Experiments:
  * [experiment1.py](experiment1.py) -- generates fake data appropriate for linear regression on uses an instance of the
    `Net` class in [SGD_nn.py](SGD_nn.py) to find the least-squares regression line for the data.
  * [experiment2.py](experiment2.py) -- runs experiment1 many times and tabulates and analyses the mean and variation
    of the slopes and standard deviations of the resulting regression lines.
