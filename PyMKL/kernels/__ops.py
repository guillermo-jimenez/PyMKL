from typing import List, Callable, Dict, Tuple, Union, Any

import math
import numpy as np
import scipy as sp
import numpy.matlib
import sak

from scipy.spatial.distance import pdist, cdist, squareform

KERNEL_LIST = ("euclidean", "euclidean_density", "categorical", "ordinal", "xcorr", "euclidean_xcorr", "default")

def euclidean_xcorr(x: np.ndarray, knn: int = 5, alpha: float = -1, **kwargs):
    x = x.copy().squeeze()
    if x.ndim == 1:
        x = x[:,None]
    if x.ndim != 2:
        raise ValueError("euclidean_xcorr kernel must take 1D or 2D inputs")

    # Obtain pairwise distances
    K_eucl,_,_ = euclidean(x,knn,alpha,**kwargs)
    K_corr,_,_ = xcorr(x,**kwargs)
    ptg = kwargs.get("proportion",0.5)
    K = ptg*K_eucl + (1-ptg)*K_corr
    var = np.var(K)

    return K,var,1


def xcorr(x: np.ndarray, **kwargs):
    x = x.copy().squeeze()
    if x.ndim == 1:
        x = x[:,None]
    if x.ndim != 2:
        raise ValueError("xcorr kernel must take 1D or 2D inputs")

    # Obtain pairwise distances
    K = (1+sak.signal.xcorr(x,maxlags=0)[0].squeeze())/2
    var = np.var(K)

    return K,var,1


def euclidean(x: np.ndarray, knn: int = 5, alpha: float = -1, **kwargs):
    x = x.copy().squeeze()
    if x.ndim == 1:
        x = x[:,None]
    if x.ndim != 2:
        raise ValueError("Euclidean kernel must take 1D or 2D inputs")

    # Get dimensions
    N = x.shape[0]

    # Obtain pairwise distances
    distances = squareform(pdist(x,metric="euclidean"))
    
    # Obtain inf-diagonal distances
    inf_distances = distances.copy()
    np.fill_diagonal(inf_distances,np.inf)
    
    # Sort these distances and retrieve the <knn>-th most similar elements for computing sigma
    inf_distances_sorted = np.sort(inf_distances,axis=0)
    sigma = np.mean(inf_distances_sorted[:min([knn,N]),:])
    
    # Obtain kernel value
    K = np.exp(alpha*(np.square(distances) / (2.*(sigma)**2.)))
    var = np.var(K)

    return K,var,sigma


def euclidean_density(x: np.ndarray, knn: int = 5, alpha: float = -1, **kwargs):
    x = x.copy().squeeze()
    if x.ndim == 1:
        x = x[:,None]
    if x.ndim != 2:
        raise ValueError("Euclidean kernel must take 1D or 2D inputs")

    # Get dimensions
    N = x.shape[0]

    # Obtain pairwise distances
    distances = squareform(pdist(x,metric="euclidean"))
    
    # Obtain inf-diagonal distances
    inf_distances = distances.copy()
    np.fill_diagonal(inf_distances,np.inf)
    
    # Sort these distances and retrieve the <knn>-th most similar elements for computing sigma
    inf_distances_sorted = np.sort(inf_distances,axis=0)
    sigma = np.mean(inf_distances_sorted[:min([knn,N]),:],0)
    sigma[sigma == 0] = np.min(sigma[sigma > 0])
    sigma = np.matlib(sigma,len(sigma),1).T
    
    # Obtain kernel value
    K = np.exp(alpha*(np.square(distances) / (2.*(sigma)**2.)))
    var = np.var(K)

    return K,var,sigma


def categorical(x: np.ndarray, *args, **kwargs):
    x = x.copy().squeeze()
    if x.ndim > 1:
        raise ValueError("Categorical kernel must take 1D inputs")

    # Count occurrences of each category    
    counts = np.bincount(x)
    unique = np.unique(x)

    # Take out zero if present (why, numpy, why)
    if counts.size > unique.size:
        counts = counts[1:]

    # Compute probability of each category in population
    prob = np.zeros((len(x),))
    for i,u in enumerate(unique):
        prob[x == u] = counts[i]/len(x)
    prob = np.matlib.repmat(prob[None,],len(x),1)

    # Compute kernel
    K = (x[:,None] == x[None,]) * (1-prob)

    if np.min(prob) > 0.05:
        K[K == 0] = np.min(prob)-0.05

    return K,1,1


def ordinal(x: np.ndarray, *args, **kwargs):
    x = x.copy().squeeze()
    # Compute kernel
    if x.ndim == 1:
        x_range = np.max(x) - np.min(x)
        K = (x_range - np.abs(x[:,None] - x[None,]))/x_range
    elif x.ndim == 2:
        distances = squareform(pdist(x,metric="euclidean"))
        x_range = np.max(distances) - np.min(distances)
        K = (x_range - distances)/x_range
    else:
        raise ValueError("Ordinal kernel must take 1D or 2D inputs")

    return K,1,1


def default(x: np.ndarray, *args, **kwargs):
    x = x.copy().squeeze()
    if x.ndim != 1:
        raise ValueError("Categorical kernel must take 1D inputs")

    # Compute kernel
    K = (x[:,None] == x[None,]).clip(min=0.9)

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


def compute_kernel(XA: np.ndarray, XB: np.ndarray = None, metric: str = "euclidean", knn: int = None, sigma: float = None) -> Tuple[np.ndarray,np.ndarray]:
    """Compute kernels from one or two sets of features through a metric. Current metric is 
    squareform(pdist(x,metric="euclidean")), but any metric that operates on a 
    matrix M ∈ [S x L], where S is the number of samples in the population and L
    is the length of the sample, is valid. Moreover, custom metrics can operate
    with features that contain samples with different lengths in a List[List[float]]
    fashion.

    Inputs:
    * XA:       np.ndarray
    * XB:       np.ndarray
    * metric:   str, metric to be used in scipy.spatial.distance.pdist (or cdist)
    * knn:      <int>, number of nearest neighbours for the computation of the sigma

    For further details, refer to https://doi.org/10.1016/j.media.2016.06.007 and to 
    https://doi.org/10.1109/TPAMI.2010.183
    """

    # Obtain pairwise distances
    if XB is None:
        distances = squareform(pdist(XA,metric=metric))
    else:
        XAB = np.concatenate((XA,XB),axis=0)
        distances = cdist(XA,XAB,metric=metric)

    if sigma is None:
        # Obtain inf-diagonal distances
        inf_distances = distances.copy()
        np.fill_diagonal(inf_distances,np.inf)
        
        # Sort these distances
        inf_distances_sorted = np.sort(inf_distances,axis=0)

        # Retrieve the <knn>-th most similar elements for computing sigma
        sigma = np.mean(inf_distances_sorted[:knn,:])

    # Obtain kernel value
    K = np.exp(-1. * (np.square(distances) / (2.*(sigma)**2.)))
    var = np.var(K)

    if XB is not None:
        K = K[:,-XB.shape[0]:]

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
