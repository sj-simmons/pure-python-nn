# nnutils.py                                                   Simmons June 2018
#
# Provides is a function that's convenient for training an instance of the Net
# class in nnff.py.

import random


#####################  Some utilility functions ################################

#pylint: disable=too-many-arguments,too-many-locals
def train(net, xss, yss, batchsize, epochs, learning_rate, prtlns=30):
  """
  A convenient function that implements a loop for training instances of Net
  in ffnn.py.  It spews out the last prtlns current losses without the cost of
  computing the current loss when you don't really need to see it's exact value.
  It also deals with the situation when the number of samples isn't divisble by
  the mini-batch size.
  """

  n_examples = len(xss)
  thresh = epochs*n_examples/batchsize-int(prtlns*batchsize/n_examples)-1

  for i in range(int(epochs * n_examples / batchsize)):
    xyss = list(zip(xss, yss))
    random.shuffle(xyss)
    xss, yss = zip(*xyss)
    total_ave_loss = 0
    for j in range(0, n_examples, batchsize):
      xss_mb = (xss + xss[:batchsize])[j: j + batchsize]
      yss_mb = (yss + yss[:batchsize])[j: j + batchsize]
      net.zero_grads()
      loss = net.learn(xss_mb, yss_mb, learning_rate)
      total_ave_loss = (total_ave_loss + loss)/2
      if i >= thresh and j > n_examples - batchsize * prtlns:
        print('current loss: {0:12.4f}'.format(total_ave_loss))
    if i <= thresh:
      print('current loss: {0:12.4f}'.format(total_ave_loss), end='\b'*26)
  return net

