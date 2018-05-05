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
  * [SGD_nn.py](SGD_nn.py): The building blocks of a feed forward neural net.
    Along with some functions to create non-linearity, the following classes are defined:
    * `Net` -- The neural net, which is built from layers (lists) of instances of the `Node` class.
    * `Node` -- essentially a list of instances of `InputLinks` along with some methods.
    * `InputLink` -- a small class with attributes `weight` and `inputNode`, instances of which connect the
      instances of Node that make up the `Net`.
    Notes: 
    * The inputs and outputs to an instance of Net are assumed to be lists of list, one list for
      each example in the data set.
    * The `Net` class currently trains using SGD (stochastic gradient descent). 
    TODO: 
    * Implement mini-batch gradient descent.
  * [PurePythonStats.py](PurePythonStats.py): This is small library of functions, written in pure Python,
    that are useful, for example, for mean_centering and normalizing data in the form of lists of lists.
