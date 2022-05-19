import numpy as np
import scipy as sp
import scipy.linalg


def to_PDM(M, eps):
    # make it symmetric and positive definite
    M = 0.5 * (M + M.transpose())
    if np.allclose(np.matrix(M),np.matrix(M).H):
        # If is Hermitian, perform scipy's eigh
        val, vec = sp.linalg.eigh(M)
    else:
        # Else, compute scipy's eig
        val, vec = sp.linalg.eig(M)

    # If any eigenvalue is sufficiently small
    if M.dtype == np.float64:
        val[val <= eps] = 5e-14  # eps
    elif M.dtype == np.float32:
        val[val <= eps] = 2.25e-7  # eps

    d = np.diag(val)
    M = vec.dot(d).dot(vec.transpose())    
    return M

def create_ivalues(numWorkers,N):
    i                       = np.arange(N,0.,-1)
    a                       = np.cumsum(i)
    t                       = np.ceil(a[-1]/numWorkers)
    ts                      = t*np.arange(1,numWorkers+1)
    mask                    = np.sum((a[:,np.newaxis] <= ts[np.newaxis,:]).astype(int),axis=1)
    dmask                   = np.concatenate(([-1],np.diff(mask)))
    ivalues                 = np.concatenate((np.where(dmask == -1)[0],[N]))
    return ivalues


