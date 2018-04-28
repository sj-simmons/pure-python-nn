# Pure_Python_Stats.py                                                     Simmons  Spring 2018
#
# This is a simple stats library in pure Python.  Most of the functions below operate on m-by-n
# arrays but those have to be represents as lists of lists, and so are called 'arrayLists';
#
#                                                                           +-       -+ 
#                                                                           |  1    2 |
# like, say, this [[1,2],[3,-4],[-5,-6]] which represents the 3-by-2 array  |  3   -4 |.
#                                                                           | -5   -6 |
#                                                                           +-       -+ 
#
# Note: Instead of this, just install and use numpy for working with arrays if you are able to.


def ones(m,n):
    """
    Return an m-by-n arrayList consisting of all ones.

    >>> ones(2,3)
    [[1, 1, 1], [1, 1, 1]]
    """
    return m * [n * [1]]


def addLists(lst1, lst2):
    """
    Return the elementwise sum of the lst1 and lst2. 

    >>> addLists([1, 2, 3],[4, 5, 6])
    [5, 7, 9]
    """
    assert len(lst1) == len(lst2), "The lists have to be the same length."
    return list(map(lambda tuple_ : tuple_[0] + tuple_[1], zip(lst1, lst2)))


def multiplyLists(lst1, lst2):
    """ Return the elementwise product of the lst1 and lst2. """

    assert len(lst1) == len(lst2), "The lists have to be the same length."
    return list(map(lambda tuple_ : tuple_[0] * tuple_[1], zip(lst1, lst2)))


def scalarMult(lst, arrList):
    """ 
    Return arrList with the first column multiplied by the first element of lst, the second 
    column mutiplied by the second element of lst, and so on.

    >>> scalarMult([2], [[1], [2], [3]])
    [[2], [4], [6]]
    >>> scalarMult([2, 3], [[1, 4], [2, 5], [3, 6]])
    [[2, 12], [4, 15], [6, 18]]
    """
    assert len(lst) == len(arrList[0]), \
         "Length of list must be the same as length of the first list in the arrayList" 
    return list(map(lambda lst2: multiplyLists(lst, lst2), arrList))
    

def add(arrList1, arrList2):
    """
    Return the element-wise different of two mxn arrayLists. 

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
    Return the element-wise different of two mxn arrayLists. 

    >>> subtract([[1, 2, 3]], [[4, 5, 6]])
    [[-3, -3, -3]]
    >>> subtract([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]])
    [[-5, -3, -1], [1, 3, 5]]
    >>> subtract([[]], [[]])
    [[]]
    """
    return add(arrList1, scalarMult([-1]*len(arrList2[0]), arrList2))


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
    """ Return a list consisting of the means of the columns of arrayList. """

    def helper(arrList):
        if len(arrList) == 1:
            return arrList[0]
        else:
            return addLists(arrList[0], helper(arrList[1:]))
    return list(map(lambda x: x / len(arrList), helper(arrList)))

 
def mean_center(arrList):
    """
    Mean-center the arrayList.
 
    Args:
    arrList - a list of lists

    Returns:
    list, arrayList  - a pair consisting of a list each entry of which is the mean of the 
                       corresponding column of arrList, a list of lists each entry of which
                       is that entry minus the mean of the column that that entry is in.. 
    >>> mean_center([[1, 2, 3], [6, 7, 8]])
    ([3.5, 4.5, 5.5], [[-2.5, -2.5, -2.5], [2.5, 2.5, 2.5]])
    >>> mean_center([[1, 2, 3], [4, 5, 6]])
    ([2.5, 3.5, 4.5], [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
    >>> mean_center([[]])
    ([], [[]])
    """
    means = columnwise_means(arrList) # a list holding the means of the columns of arrList
    arrList = subtract(arrList, scalarMult(means, ones(len(arrList), len(arrList[0]))))
    return means, arrList

        
def normalize(arrList):
    """
    Normalize the arrayList.
 
    Args:
    arrayList - a list of lists

    Returns:
    list, arrayList  - a pair consisting of a list each entry of which is the standard 
                       deviation of the corresponding column of arrList, a list of lists 
                       each entry of which is that entry divided by the standard deviation 
                       of the column that that entry is in.

    >>> normalize([[1, 2, 3], [6, 7, 8]]) # doctest:+ELLIPSIS 
    ([2.5, 2.5, 2.5],...
    >>> normalize([[1, 2, 3], [1, 7, 3]]) # doctest:+ELLIPSIS 
    ([1, 2.5, 1],...
    >>> normalize([[]])
    ([], [[]])
    """
    _, centered = mean_center(arrList)
    centered_squared = multiply(centered, centered)
    stdevs = list(map(lambda x: x**0.5, columnwise_means(centered_squared)))
    stdevs = list(map(lambda x: 1 if x == 0 else x, stdevs))
    inverses = list(map(lambda x: 1/x, stdevs))
    return stdevs, multiply(scalarMult(inverses, ones(len(arrList), len(arrList[0]))), arrList) 


if __name__ == "__main__":
    import doctest
    doctest.testmod()
