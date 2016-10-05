import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # if the margin_ij > 0, X[i] has to be added to dW and subtracted from
        # dW[:, correct class of i (i.e. y[i])],
        # because s_nc = \Sigma_{d = 1 ~ D} x_id * w_dc
        # and s_ny[i] = \Sigma_{d = 1 ~ D} x_id * w_dy[i]
        # margin = s_nc - s_ny[i] + 1
        # so for each w_dc, adding x_id
        #  , for each w_dy[i], subtracting x_id
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  right_class_scores = scores[range(len(y)), y]

  margins = np.maximum(np.array(scores - np.matrix(right_class_scores).T) + 1, 0)
  margins[range(num_train), y] = 0
  loss += np.sum(margins) / num_train
  loss += reg * 0.5 * np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  tmp = np.zeros(margins.shape)
  tmp[margins > 0] = 1
  num_pos_data = np.sum(margins > 0, axis = 1)
  tmp[range(margins.shape[0]), y] -= num_pos_data
  # now tmp_nc means how many times X[n,:] has to be summed(or subtracted)
  # n-th column of X.T is X[n,:]
  # X.T.dot(tmp[:,c]) is linear combination of {X[n,:]| n = 1 ... N }
  # these compose colums of X.T.dot(tmp)
  dW = X.T.dot(tmp) / num_train + reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
