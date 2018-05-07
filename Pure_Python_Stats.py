# Pure_Python_Stats.py                                                     Simmons  Spring 2018
#
# This is a simple 'tensor-like' stats library in pure Python 3.  Most of the functions below
# operate on m-by-n arrays but those have to be represents as lists of lists, and so are called
# 'arrayLists';
#
#                                                                             +-       -+
#                                                                             |  1    2 |
# like, say, this: [[1,2],[3,-4],[-5,-6]], which represents the 3-by-2 array  |  3   -4 |.
#                                                                             | -5   -6 |
#                                                                             +-       -+
#
# Note: Instead of using this in practice, just install and use numpy for working with arrays
#       if you are able to.

from functools import reduce

def ones(m,n):
    """
    Return an m-by-n arrayList consisting of all ones.

    >>> ones(2,3)
    [[1, 1, 1], [1, 1, 1]]
    """
    return m * [n * [1]]

def addLists(lst1, lst2):
    """
    Return the elementwise sum of lst1 and lst2.

    >>> addLists([1, 2, 3],[4, 5, 6])
    [5, 7, 9]
    """
    assert len(lst1) == len(lst2), "The lists have to be the same length."
    return list(map(lambda tuple_ : tuple_[0] + tuple_[1], zip(lst1, lst2)))

def multiplyLists(lst1, lst2):
    """ Return the elementwise product of lst1 and lst2. """

    assert len(lst1) == len(lst2), "The lists have to be the same length."
    return list(map(lambda tuple_ : tuple_[0] * tuple_[1], zip(lst1, lst2)))

def divideLists(lst1, lst2):
    """
    Return the elementwise quotient of lst1 and lst2.

    >>> divideLists([1, 2, 3],[4, 5, 6])
    [0.25, 0.4, 0.5]
    >>> divideLists([1, 2, 3],[4, 0, 6])
    Traceback (most recent call last):
     ...
    AssertionError: Second list has entry equal to zero.
    """

    assert len(lst1) == len(lst2), "The lists have to be the same length."
    assert reduce(lambda x,y: x*y, lst2) != 0, "Second list has entry equal to zero."
    return list(map(lambda tuple_ : tuple_[0] / tuple_[1], zip(lst1, lst2)))

def dotLists(lst1, lst2):
    """
    Return the dot product of the vectors defined by lst1 and lst2.

    >>> dotLists([1, 2, 3], [4, 5, 6])
    32
    """
    assert len(lst1) == len(lst2), "The lengths of the lists should be the same."
    return reduce(lambda x,y: x+y, [pair[0] * pair[1] for pair in zip(lst1, lst2)])

def scalarMultCols(lst, arrList):
    """
    Return arrList with the first column scalar multiplied by the first element of lst, the 
    second column scalar multiplied by the second element of lst, and so on.

    >>> scalarMultCols([2], [[1], [2], [3]])
    [[2], [4], [6]]
    >>> scalarMultCols([2, 3], [[1, 4], [2, 5], [3, 6]])
    [[2, 12], [4, 15], [6, 18]]
    """
    assert len(lst) == len(arrList[0]), \
         "Length of list must be the same as length of the first list in the arrayList"
    return list(map(lambda lst2: multiplyLists(lst, lst2), arrList))


def add(arrList1, arrList2):
    """
    Return the element-wise sum of two mxn arrayLists.

    >>> add([[1, 2, 3]], [[4, 5, 6]])
    [[5, 7, 9]]
    >>> add([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]])
    [[7, 7, 7], [7, 7, 7]]
    >>> add([[]], [[]])
    [[]]
    """
    assert len(arrList1[0]) == len(arrList1[0]) and len(arrList1) == len(arrList1),\
        "arrayLists must be list of lists, one with the same number of rows and columns\
         as the other"
    if len(arrList1) == 1:
        return [[eltpair[0] + eltpair[1] for eltpair in zip(arrList1[0], arrList2[0])]]
    else:
        return [add([rowpair[0]], [rowpair[1]])[0] for rowpair in zip(arrList1, arrList2)]


def subtract(arrList1, arrList2):
    """
    Return the element-wise difference of two mxn arrayLists.

    >>> subtract([[1, 2, 3]], [[4, 5, 6]])
    [[-3, -3, -3]]
    >>> subtract([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]])
    [[-5, -3, -1], [1, 3, 5]]
    >>> subtract([[]], [[]])
    [[]]
    """
    return add(arrList1, scalarMultCols([-1]*len(arrList2[0]), arrList2))


def multiply(arrList1, arrList2):
    """
    Return the element-wise product of two mxn arrayLists.

    >>> multiply([[1, 2, 3]], [[4, 5, 6]])
    [[4, 10, 18]]
    >>> multiply([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]])
    [[6, 10, 12], [12, 10, 6]]
    >>> multiply([[]], [[]])
    [[]]
    """
    assert len(arrList1[0]) == len(arrList1[0]) and len(arrList1) == len(arrList1),\
        "arrayLists must be list of lists, one with the same number of rows and columns\
         as the other"
    if len(arrList1) == 1:
        return [[eltpair[0] * eltpair[1] for eltpair in zip(arrList1[0], arrList2[0])]]
    else:
        return [multiply([rowpair[0]], [rowpair[1]])[0] for rowpair in zip(arrList1, arrList2)]


def columnwise_means(arrList):
    """ 
    Return a list consisting of the means of the columns of arrayList. 
   
    >>> columnwise_means([[1, 2, 3], [4, 5, 6]])
    [2.5, 3.5, 4.5]
    """

    import sys
    sys.setrecursionlimit(len(arrList)+100)  # quick fix for recursion depth error

    def helper(arrList):
        if len(arrList) == 1:
            return arrList[0]
        else:
            return addLists(arrList[0], helper(arrList[1:]))
    return list(map(lambda x: x / len(arrList), helper(arrList)))


def mean_center(arrList):
    """
    Mean-center the columns of the arrayList.

    Args:
        arrList (a list of lists of numbers)

    Returns:
        list, list: A pair consisting of a list each entry of which is the mean of the
                    corresponding column of arrList, a list of lists each entry of which
                    is that entry minus the mean of the column that that entry is in.

    >>> mean_center([[1, 2, 3], [6, 7, 8]])
    ([3.5, 4.5, 5.5], [[-2.5, -2.5, -2.5], [2.5, 2.5, 2.5]])
    >>> mean_center([[1, 2, 3], [4, 5, 6]])
    ([2.5, 3.5, 4.5], [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
    >>> mean_center([[]])
    ([], [[]])
    """
    means = columnwise_means(arrList) # a list holding the means of the columns of arrList
    arrList = subtract(arrList, scalarMultCols(means, ones(len(arrList), len(arrList[0]))))
    return means, arrList

def normalize(arrList):
    """
    Normalize the arrayList, meaning divide each column by its standard deviation if that
    standard deviation is nonzero, and leave the column unmodified if it's standard deviation
    is zero.

    Note: Columns 

    Args:
        arrList (a list of lists of numbers)

    Returns:
        list, list: A pair consisting of a list each entry of which is the standard deviation
                    of the corresponding column of arrList, and an arrayList (a list of lists)
                    each column of which is that column divided by the standard deviation of 
                    the column that column if that standard deviation is nonzero.  Columns
                    with zero standard deviation are left unchanged.

    >>> normalize([[1, 2, 3], [6, 7, 8]]) # doctest:+ELLIPSIS
    ([2.5, 2.5, 2.5],...
    >>> normalize([[1, 2, 3], [1, 7, 3]]) # doctest:+ELLIPSIS
    ([0.0, 2.5, 0.0],...
    >>> normalize([[]])
    ([], [[]])
    """
    _, centered = mean_center(arrList)
    centered_squared = multiply(centered, centered)
    stdevs = list(map(lambda x: x**0.5, columnwise_means(centered_squared)))
    nonzero_stdevs = list(map(lambda x: 1 if x == 0 else x, stdevs))
    inverses = list(map(lambda x: 1/x, nonzero_stdevs))
    return stdevs, scalarMultCols(inverses, arrList)

def un_center(means, arrList):
    """
    Return an arrayList with ith column un_mean_centered by scalar means[i]

    Args:
        means: A list of numbers (should be the list output by mean_center).
        arrList: A list of list of numbers that is the (mean-centered) data.
    Returns:
        list: A list of list of numbers.
    """
    return add(arrList, scalarMultCols(means, ones(len(arrList), len(arrList[0]))))

def un_normalize(stdevs, arrList):
    """
    Return an arrayList with ith column multiplied by scalar stdevs[i] if stdevs[i] is not zero,
    and unmodified if it is zero.

    Args:
        stdevs: A list of numbers (should be the list output by normalize).
        arrList: A list of list of numbers that is the (normalized) data.
    Returns:
        list: A list of list of numbers.

    >>> un_normalize([0.5, 2],[[1, 2], [3,4]])
    [[0.5, 4], [1.5, 8]]
    >>> un_normalize([0.0, 2],[[1, 2], [3,4]])
    [[1, 4], [3, 8]]
    """
    stdevs = list(map(lambda x: x if x != 0.0 else 1, stdevs))
    return scalarMultCols(stdevs, arrList)

def un_normalize_slopes(lst, xstdevs, ystdevs):
    """ Helper function for un_map_weights below. """

    assert len(lst) == len(xstdevs) and len(ystdevs) == 1,\
     "First and second list have to be the same length; third currently has to be length 1."+\
     " The sizes are: " + str(len(lst)) + ", " +  str(len(xstdevs)) + ", " +  str(len(ystdevs))

    xstdevs = list(map(lambda x: x if x != 0.0 else 1, xstdevs))
    return multiplyLists(lst, divideLists(ystdevs * len(xstdevs), xstdevs))

def un_map_weights(weights, xmeans, xstdevs, ymeans, ystdevs):
    """
    Shift weights to those of model trained on mean-centered and normalized data to the
    weights of the corresponding un-mean-centered, un-normalized model.

    Note: This is only useful for getting the coefficients for writing the regression plane
          in _linear_ regression.

    Args:
        weights: A list of weights (numbers) with first entry corresponding to the bias.
        xmeans: A list of the x-variable means (numbers).
        xstdevs: A list of the x-variable standard deviations (numbers).
        ymeans: A list of the y-variable means (numbers).
        ystdevs: A list of the y-variable standard deviations (numbers).
    Returns:
        list: The new weights with the first entry corresponding to the bias.

    Todo:
        Generalize this to handle more than one y variable.
    """
    assert len(xmeans) == len(xstdevs) and \
           len(ystdevs) == len(ystdevs) == 1,\
           len(weights) == len(xmeans)

    slopes = un_normalize_slopes(weights, xstdevs, ystdevs)
    intercept = ymeans[0] - dotLists(slopes, xmeans)
    return [intercept] + slopes


if __name__ == '__main__':
    import doctest
    doctest.testmod()
