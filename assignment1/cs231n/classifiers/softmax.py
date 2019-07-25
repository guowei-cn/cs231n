from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    num_data = X.shape[0]
    num_class = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    for i in range(num_data):
        scores = X[i].dot(W) #(1, D)(D, C) = (1, C)
        shift_scores = scores - np.max(scores)
        softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores))
        loss += - np.log(softmax_output[y[i]])
        for j in range(num_class):
            if j == y[i]:
                dW[:,j] += (softmax_output[j] - 1) * X[i] # (1, C) (1,D)
            else:
                dW[:,j] += softmax_output[j] * X[i]
    #pass
    loss = loss/num_data + reg * np.sum(W * W)
    dW = dW/num_data + 2 * reg * W 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_data = X.shape[0]
    num_class = W.shape[1]
    
    scores = X.dot(W) # (N,C)
    shift_scores = scores - np.amax(scores, axis = 1).reshape(-1, 1)
    softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1, 1)# (N, C)
    loss = -np.sum(np.log(softmax_output[range(num_data), list(y)]))/num_data + reg * np.sum(W * W)
    
    copy_softmax_output = np.copy(softmax_output)
    copy_softmax_output[range(num_data), list(y)] -= 1
    dW = X.T.dot(copy_softmax_output)/num_data + 2 * reg * W
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
