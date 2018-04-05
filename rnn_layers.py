from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None

    temp_next_h = np.dot(prev_h, Wh) + (np.dot(x, Wx) + b)
    next_h = np.tanh(temp_next_h)
    cache = (prev_h, x, Wx, Wh, next_h)

    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None

    prev_h, x, Wx, Wh, next_h = cache
#     temp = np.arctanh(next_h)
    dtemp = np.multiply((1 - np.square(next_h)), dnext_h)  # derivative of tanh in its own form
    dprev_h = np.dot(dtemp, Wh.T)
    dWh = np.dot(prev_h.T, dtemp)
    dx = np.dot(dtemp, Wx.T)
    dWx = np.dot(x.T, dtemp)
    db = np.sum(dtemp, axis = 0)    

    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None

    N,T,D = x.shape
    H, = b.shape
    prev_h = h0
    h = np.zeros((N,T,H))
    for i in range(T):
        xt = x[:,i,:]
        next_h, _ = rnn_step_forward(xt, prev_h, Wx, Wh, b)
        prev_h = next_h
        h[:,i,:] = next_h
    cache = (x, h0, Wx, Wh, b,h)

    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None

    x, h0, Wx, Wh, b, h = cache
    N, T, H = dh.shape
    _, _, D = x.shape

    next_h = h[:,T-1,:]
    
    dprev_h = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx= np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H,))
    
    for t in reversed(range(T)):
        xt = x[:,t,:]      
        dnext_h = dh[:,t,:] + dprev_h
        next_h = h[:,t,:]
        
        if t == 0:
            prev_h = h0
        else:
            prev_h = h[:,t-1,:]

        cache1 = (prev_h, xt, Wx, Wh, next_h) 

        dx[:,t,:], dprev_h, dWxt, dWht, dbt = rnn_step_backward(dnext_h,cache1)
        
        dWx += dWxt
        dWh += dWht
        db += dbt
        
    dh0 = dprev_h

    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    out = W[x]   
    cache = (x,W)

    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None

    x,W = cache
    dW = np.zeros_like(W)
    np.add.at(dW,x,dout)

    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None

    _,H = prev_h.shape
    temp = x.dot(Wx) + prev_h.dot(Wh) + b
    
#     print(temp.shape)
    gate_i = sigmoid(temp[:,:H])
    gate_f = sigmoid(temp[:,H:2*H])
    gate_o = sigmoid(temp[:,2*H:3*H])
    gate_g = np.tanh(temp[:,3*H:])
    
    next_c = gate_f * prev_c + gate_i * gate_g
    temp = np.tanh(next_c)
    next_h = gate_o * temp
    
    cache = (x, Wx, Wh, b, prev_h, prev_c, gate_i, gate_f, gate_o, gate_g, temp)

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None

    x, Wx, Wh, b, prev_h, prev_c, gate_i, gate_f, gate_o, gate_g, temp = cache
    dx = np.zeros_like(x)
    dprev_h = np.zeros_like(prev_h)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)    

    dtemp = gate_o * dnext_h
    dsplit = (1 - np.square(temp)) * dtemp
    dnext_c += dsplit
    dprev_c = gate_f * dnext_c
    
    dgate_o_a = dnext_h * temp
    dgate_f_a = dnext_c * prev_c
    dgate_i_a = dnext_c * gate_g
    dgate_g_a = dnext_c * gate_i
    
    dgate_o = gate_o * (1 - gate_o) * dgate_o_a
    dgate_f = gate_f * (1 - gate_f) * dgate_f_a
    dgate_i = gate_i * (1 - gate_i) * dgate_i_a
    dgate_g = (1 - np.square(gate_g)) * dgate_g_a
    
    
    dstacked = np.hstack((dgate_i,dgate_f,dgate_o,dgate_g))
    
    dWh = prev_h.T.dot(dstacked)
    dprev_h = dstacked.dot(Wh.T)    
    db = np.sum(dstacked,axis=0)
    dWx = x.T.dot(dstacked)
    dx = dstacked.dot(Wx.T)


    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None

    N,T,D = x.shape
    _,H = h0.shape
    prev_h = h0
    h = np.zeros((N,T,H))
    prev_c = np.zeros_like(h0)
    cache = []
    for i in range(T):
        xt = x[:,i,:]
        
        next_h,next_c,temp_cache = lstm_step_forward(xt, prev_h, prev_c, Wx, Wh, b)
        h[:,i,:]= next_h
        prev_h = next_h
        prev_c = next_c
        cache.append(temp_cache)

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None

    N,T,H = dh.shape
    
    D = cache[T-1][0].shape[1] #gather D cache[any][0] contains x at that timestep
#     print(D)
    
    
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H,))
    dprev_h = np.zeros((N,H))
    dprev_c = np.zeros((N,H))
    
    for t in reversed(range(T)):
        dnext_h = dh[:,t,:] + dprev_h
        dnext_c = dprev_c        
        

        dx[:,t,:], dprev_h, dprev_c, dWxt, dWht, dbt = lstm_step_backward(dnext_h,dnext_c,cache[t])
#         for i in [dx[:,t,:], dprev_h, dprev_c, dWxt, dWht, dbt]:
#             print(i.shape)
#         break
        dWx,dWh,db = dWx+dWxt, dWh+dWht, db+dbt
#         dx[:,t,:] = dxt
        
    dh0 = dprev_h

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
