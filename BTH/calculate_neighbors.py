import os
import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.io
import h5py

def pdist2(X, Y, metric):
  metric = metric.lower()
  if metric == 'sqeuclidean':
    X = X.astype('float64')
    Y = Y.astype('float64')
    nx = X.shape[0]
    ny = Y.shape[0]
    XX = np.tile((X**2).sum(1),(ny,1)).T
    YY = np.tile((Y**2).sum(1),(nx,1))
    XY = X.dot(Y.T)
    sqeuc = XX + YY - 2*XY
    # Make negatives equal to zero. This arises due to floating point
    # precision issues. Negatives will be very close to zero (IIRC around
    # -1e-10 or maybe even closer to zero). Any better fix? you exhibited the
    #  floating point issue on two machines using the same code and data,
    # but not on a third. the inconsistent occurrence of the issue could
    # possibly be due to differences in numpy/blas versions across machines.
    return np.clip(sqeuc, 0, np.inf)
  elif metric == 'hamming':
    # scipy cdist supports hamming distance, but is twice as slow as yours
    # (even before multiplying by dim, and casting as int), possibly because
    # it supports non-booleans, but I'm not sure...
    # looping over data points in X and Y, and calculating hamming distance
    # to put in a hamdis matrix is too slow. this vectorized solution works
    # faster. separately, the matlab solution that uses compactbit is
    # approximately 8x more memory efficient, since 8 bits are required here
    # for each 0 or 1.
    hashbits = X.shape[1]
    Xint = (2 * X.astype('int8')) - 1
    Yint = (2 * Y.astype('int8')) - 1
    hamdis = hashbits - ((hashbits + Xint.dot(Yint.T)) / 2)
    return hamdis
  else:
    valerr = 'Unsupported Metric: %s' % (metric,)
    raise ValueError(valerr)

def ZZ(data, anchors, nnanchors, sigma):
    n = data.shape[0]
    m = anchors.shape[0]
    # pdb.set_trace()
    
    # tried using for loops. too slow.
    sqdist = pdist2(data, anchors, 'sqeuclidean')
    val = np.zeros((n, nnanchors))
    pos = np.zeros((n, nnanchors), dtype=np.int)
    for i in range(nnanchors):
      pos[:,i] = np.argmin(sqdist, 1)
      val[:,i] = sqdist[np.arange(len(sqdist)), pos[:,i]]
      sqdist[np.arange(n), pos[:,i]] = float('inf')
    
    # would be cleaner to calculate sigma in its own separate method,
    # but this is more efficient
    if sigma is None:
      dist = np.sqrt(val[:,nnanchors-1])
      sigma = np.mean(dist) / np.sqrt(2)
    
    # Next, calculate formula (2) from the paper
    # this calculation differs from the matlab. In the matlab, the RBF
    # kernel's exponent only has sigma^2 in the denominator. Here,
    # 2 * sigma^2. This is accounted for when auto-calculating sigma above by
    #  dividing by sqrt(2)
    
    # Work in log space and then exponentiate, to avoid the floating point
    # issues. for the denominator, the following code avoids even more
    # precision issues, by relying on the fact that the log of the sum of
    # exponentials, equals some constant plus the log of sum of exponentials
    # of numbers subtracted by the constant:
    #  log(sum_i(exp(x_i))) = m + log(sum_i(exp(x_i-m)))
    
    c = 2 * np.power(sigma,2) # bandwidth parameter
    exponent = -val / c       # exponent of RBF kernel
    shift = np.amin(exponent, 1, keepdims=True)
    denom = np.log(np.sum(np.exp(exponent - shift), 1, keepdims=True)) + shift
    val = np.exp(exponent - denom)
    
    Z = np.zeros((n,m),dtype='float32')
    for i in range(nnanchors):
      Z[np.arange(n), pos[:,i]] = val[:,i]
    # Z = scipy.sparse.csr_matrix(Z)
    
    return Z, sigma, pos

def get_neighbor(data, anchors, nnanchors):
    n = data.shape[0]
    m = anchors.shape[0]
    # tried using for loops. too slow.
    sqdist = pdist2(data, anchors, 'sqeuclidean')
    val = np.zeros((n, nnanchors))
    pos = np.zeros((n, nnanchors), dtype=np.int)
    for i in range(nnanchors):
      pos[:,i] = np.argmin(sqdist, 1)
    return pos
