import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  #scores = X.dot(W)
  #exp_scores = np.exp(scores)
  #log_sum_exp = np.log(np.sum(scores, axis=1))
  #fi = scores[range(scores.shape[0]), y]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss = 0
  for n in range(num_train):
    sum_exp = 0
    correct_class_score = X[n].dot(W[:,y[n]])
    scores = np.zeros((num_classes,))
    for c in range(num_classes):
        scores[c] = X[n].dot(W[:,c])
        sum_exp += np.exp(scores[c])
        if y[n] == c:
            dW[:,c] -= X[n]
    for c in range(num_classes):
        dW[:,c] += np.exp(scores[c]) / sum_exp * X[n]

    loss += -correct_class_score + np.log(sum_exp)

  loss /= num_train
  dW /= num_train
  # Add regularization to the loss
  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  dscores = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.matrix(np.amax(scores, axis=1)).T
  scores = np.array(scores)
  scores_exp = np.exp(scores)
  sum_exp = scores_exp.sum(axis=1)

  dscores = scores_exp / np.matrix(sum_exp).T
  dscores[range(num_train), y] -= 1

  log_sum_exp = np.log(sum_exp)
  correct_scores = scores[range(num_train), y]

  loss += np.sum(log_sum_exp) - np.sum(correct_scores)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW = X.T.dot(dscores)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

