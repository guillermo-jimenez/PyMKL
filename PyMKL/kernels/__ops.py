from typing import List, Callable, Dict, Tuple, Union, Any

import math
import numpy as np
import scipy as sp
import numpy.matlib
import sak

from scipy.spatial.distance import pdist, cdist, squareform

KERNEL_LIST = ("euclidean", "euclidean_density", "categorical", "ordinal", "xcorr", "euclidean_xcorr", "default")

def euclidean_xcorr(x: np.ndarray, y: np.ndarray = None, knn: int = None, alpha: float = -1, maxlags: int = 0, **kwargs):
    # Obtain pairwise distances
    K_eucl,_,_ = euclidean(x,y,knn,alpha,**kwargs)
    K_corr,_,_ = xcorr(x,y,**kwargs)
    ptg = kwargs.get("proportion",0.5)
    K = ptg*K_eucl + (1-ptg)*K_corr
    var = np.var(K)

    return K,var,1


def xcorr(x: np.ndarray, y: np.ndarray = None, **kwargs):
    x = x.copy().squeeze()
    if x.ndim == 1:
        x = x[:,None]
    if x.ndim != 2:
        raise ValueError("xcorr kernel must take 1D or 2D inputs")

    # If y is None, copy x
    if y is None:
        y = x.copy()
    else:
        y = y.copy().squeeze()
        if y.ndim == 1:
            y = y[:,None]
        if y.ndim != 2:
            raise ValueError("euclidean_xcorr kernel must take 1D or 2D inputs")

    # Obtain pairwise distances
    K = (1+sak.signal.xcorr(x,y,maxlags=0)[0].squeeze())/2
    var = np.var(K)

    return K,var,1


def euclidean(x: np.ndarray, y: np.ndarray = None, knn: int = None, alpha: float = -1, **kwargs):
    x = x.copy().squeeze()
    if x.ndim == 1:
        x = x[:,None]
    if x.ndim != 2:
        raise ValueError("Euclidean kernel must take 1D or 2D inputs")

    # If y is None, copy x
    if y is None:
        use_pdist = True
        y = x.copy()
    else:
        y = y.copy().squeeze()
        if y.ndim == 1:
            y = y[:,None]
        if y.ndim != 2:
            raise ValueError("euclidean_xcorr kernel must take 1D or 2D inputs")
        use_pdist = False

    # Get dimensions
    N = x.shape[0]

    # Apply default number of nearest neighbours
    if knn is None:
        knn = math.floor(np.sqrt(N))

    # Obtain pairwise distances
    if use_pdist:
        distances = squareform(pdist(x,metric="euclidean"))
    else:
        distances = cdist(x,y,metric="euclidean")

    # Obtain inf-diagonal distances
    inf_distances = distances.copy()
    inf_distances[inf_distances < np.finfo(distances.dtype).eps] = np.inf

    # Sort these distances and retrieve the <knn>-th most similar elements for computing sigma
    inf_distances_sorted = np.sort(inf_distances,axis=0)
    sigma = np.mean(inf_distances_sorted[:min([knn,N]),:])

    # Obtain kernel value
    K = np.exp(alpha*(np.square(distances) / (2.*(sigma)**2.)))
    var = np.var(K)

    return K,var,sigma


def euclidean_density(x: np.ndarray, y: np.ndarray = None, knn: int = None, alpha: float = -1, **kwargs):
    x = x.copy().squeeze()
    if x.ndim == 1:
        x = x[:,None]
    if x.ndim != 2:
        raise ValueError("Euclidean kernel must take 1D or 2D inputs")

    # If y is None, copy x
    if y is None:
        use_pdist = True
        y = x.copy()
    else:
        y = y.copy().squeeze()
        if y.ndim == 1:
            y = y[:,None]
        if y.ndim != 2:
            raise ValueError("euclidean_xcorr kernel must take 1D or 2D inputs")
        use_pdist = False

    # Get dimensions
    N = x.shape[0]

    # Apply default number of nearest neighbours
    if knn is None:
        knn = math.floor(np.sqrt(N))

    # Obtain pairwise distances
    if use_pdist:
        distances = squareform(pdist(x,metric="euclidean"))
    else:
        distances = cdist(x,y,metric="euclidean")

    # Obtain inf-diagonal distances
    inf_distances = distances.copy()
    inf_distances[inf_distances < np.finfo(distances.dtype).eps] = np.inf

    # Sort these distances and retrieve the <knn>-th most similar elements for computing sigma
    inf_distances_sorted = np.sort(inf_distances,axis=0)
    sigma = np.mean(inf_distances_sorted[:min([knn,N]),:],0)
    sigma[sigma == 0] = np.min(sigma[sigma > 0])
    sigma = np.matlib.repmat(sigma,N,1)

    # Obtain kernel value
    K = np.exp(alpha*(np.square(distances) / (2.*(sigma)**2.)))
    var = np.var(K)

    return K,var,sigma


def categorical(x: np.ndarray, y: np.ndarray = None, *args, **kwargs):
    x = x.copy().squeeze()
    if x.ndim > 1:
        raise ValueError("Categorical kernel must take 1D inputs")
    # If y is None, copy x
    if y is None:
        y = x.copy()
    else:
        y = y.copy().squeeze()

    # Count occurrences of each category    
    counts_x = np.bincount(x)
    unique_x = np.unique(x)
    counts_y = np.bincount(y)
    unique_y = np.unique(y)

    # Take out zero if present (why, numpy, why)
    if counts_x.size > unique_x.size:
        counts_x = counts_x[1:]
    if counts_y.size > unique_y.size:
        counts_y = counts_y[1:]

    # Compute probability of each category in population
    prob_x,prob_y = np.zeros((len(x),)),np.zeros((len(y),))
    for i,u in enumerate(unique_x):
        prob_x[x == u] = counts_x[i]/len(x)
    for i,u in enumerate(unique_y):
        prob_y[y == u] = counts_y[i]/len(y)
    prob = np.sqrt((prob_x[:,None] * prob_y[None,:]))

    # Compute kernel
    K = (x[:,None] == y[None,]) * (1-prob)

    if np.min(prob) > 0.05:
        K[K == 0] = np.min(prob)-0.05

    return K,1,1


def ordinal(x: np.ndarray, y: np.ndarray = None, *args, **kwargs):
    x = x.copy().squeeze()
    # If y is None, copy x
    if y is None:
        y = x.copy()
    else:
        y = y.copy().squeeze()
    if x.ndim != y.ndim:
        raise ValueError("x and y vectors have different depth")

    # Compute kernel
    if x.ndim == 1:
        x_range = np.max(np.concatenate((x,y))) - np.min(np.concatenate((x,y)))
        distances = np.abs(x[:,None] - y[None,])
    elif x.ndim == 2:
        distances = cdist(x,y,metric="euclidean")
        x_range = np.max(distances) - np.min(distances)
    else:
        raise ValueError("Ordinal kernel must take 1D or 2D inputs")

    K = (x_range - distances)/x_range

    return K,1,1


def default(x: np.ndarray, y: np.ndarray = None, *args, **kwargs):
    x = x.copy().squeeze()
    if x.ndim != 1:
        raise ValueError("Categorical kernel must take 1D inputs")

    # If y is None, copy x
    if y is None:
        y = x.copy()
    else:
        y = y.copy().squeeze()

    if x.ndim != y.ndim:
        raise ValueError("x and y vectors have different depth")

    # Compute kernel
    K = (x[:,None] == y[None,]).clip(min=0.9)

    return K,1,1


def kernel_stack(X: Union[List[np.ndarray],Dict[Any,np.ndarray]], kernel: Union[str,List[str]] = "euclidean", knn: int = None, alpha: float = -1, return_sigmas: bool = False) -> Tuple[np.ndarray,np.ndarray]:
    """Obtain kernels from list of features through a metric. Current metric is 
    squareform(pdist(x,metric="euclidean")), but any metric that operates on a 
    matrix M ∈ [S x L], where S is the number of samples in the population and L
    is the length of the sample, is valid. Moreover, custom metrics can operate
    with features that contain samples with different lengths in a List[List[float]]
    fashion.

    Inputs:
    * X:        <List[np.ndarray]> (or <List[List[float]]>) if custom metric is provided
    * metric:   <Callable>, so that it takes as input a <np.ndarray> or a <List[float]>, 
                according to what was provided in X.
    * knn:      <int>, number of nearest neighbours for the computation of the sigma

    For further details, refer to https://doi.org/10.1016/j.media.2016.06.007 and to 
    https://doi.org/10.1109/TPAMI.2010.183
    """

    # Retrieve dimensions
    M = len(X) # Number of different features to work with
    if isinstance(X,List):
        N = X[0].shape[0] # Number of samples in the population
    elif isinstance(X,Dict):
        keys = list(X)
        N = X[keys[0]].shape[0]
    
    # Apply default number of nearest neighbours
    if knn is None:
        knn = math.floor(np.sqrt(N))

    # Check kernel in valid kernels
    per_kernel = False
    if isinstance(kernel, str):
        if kernel not in KERNEL_LIST:
            raise ValueError(f"Valid kernels are: {KERNEL_LIST}")
        kernel = eval(kernel)
        per_kernel = False
    elif isinstance(kernel, (list, tuple)):
        for k in kernel:
            if k not in KERNEL_LIST:
                raise ValueError(f"Valid kernels are: {KERNEL_LIST}. Provided kernel: {k}")
        kernel = [eval(k) for k in kernel]
        assert len(kernel) == len(X), "Invalid kernel configuration. Must be of the same size of X"
        per_kernel = True
    else:
        raise ValueError(f"Kernel type not supported. Valid kernels are: {KERNEL_LIST} or a list of them")
        
    # Create matrix
    K = np.zeros((M, N, N), dtype='float64')
    var = np.zeros((M,), dtype='float64')
    sigmas = np.zeros((M,), dtype='float64')

    # Compute pairwise distances
    for m,k in enumerate(X):
        # Retrieve the feature according to its input data type
        if isinstance(X,List):
            feature = X[m] 
        elif isinstance(X,Dict):
            feature = X[k]

        # Obtain kernel value
        if per_kernel:
            K[m],var[m],sigmas[m] = kernel[m](feature,knn=knn,alpha=alpha)
        else:
            K[m],var[m],sigmas[m] = kernel(feature,knn=knn,alpha=alpha)

    if return_sigmas:
        return K, var, sigmas
    else:
        return K, var


def get_W_and_D(K: np.ndarray, var: np.ndarray, knn: int = None):
    # Get number of features and samples
    M = K.shape[0]
    N = K.shape[1]
    
    # Apply default number of nearest neighbours
    if knn is None:
        knn = math.floor(np.sqrt(N))
        
    # Get minimum variance
    min_var = np.min(var)
    
    # Copy the kernels tensor to avoid modifying it
    aux = np.copy(K)
    
    # Normalize features by variance
    for m in range(M):
        alfa_m = var[m]*1.0/min_var
        aux[m] = np.power(aux[m], 1/alfa_m)
        
    # Global affinity matrix: mean of the normalized affinity matrices
    W = np.sum(aux, axis=0) / M  
    sparse_W = np.zeros_like(W, dtype='float64')
    for i,r in enumerate(W):
        s = r.argsort()[::-1]
        ind_min = s[knn + 1:]
        r[ind_min] = 0
        sparse_W[i] = r
    sparse_W = np.maximum(sparse_W, sparse_W.transpose())

    # Diagonal matrix - in each diagonal element is the sum of the corresponding row
    D = np.sum(sparse_W, axis=0)[:,None]

    return sparse_W, D
