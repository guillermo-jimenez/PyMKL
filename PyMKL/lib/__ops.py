"""
Filename: libPyMKL.py
Interface with c functions:

    (1) void __computeENERGY(double *K_tot_ALL, double *betas, double *A, double *W, double *Diag,                               int N, int NA, int m, double output[2], int i1, int i2);
    (2) void    __computeSWA(double *K_tot_ALL,                double *A, double *W, double *Diag, double *S_W_A, double *S_D_A, int N, int NA, int m,                   int i1, int i2);
    (3) void    __computeSWB(double *K_tot_ALL, double *betas,            double *W, double *Diag, double *S_W_B, double *S_D_B, int N,         int m,                   int i1, int i2);

"""

import numpy.ctypeslib
import ctypes
import os
import os.path
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.spatial
import scipy.spatial.distance


# Load the library as libPyMKL.
# Why the underscore (_) in front of libPyMKL below?
# To mimimise namespace pollution -- see PEP 8 (www.python.org).
libPyMKL                         = numpy.ctypeslib.load_library(os.path.join(os.path.dirname(__file__),'libPyMKL.so'), '.')

# COMPUTE ENERGY
# void __computeENERGY(double *K_tot_ALL, double *betas, double *A, double *W, double *Diag,                               int N, int NA, int m, double output[2], int i1, int i2);
libPyMKL.computeENERGY.argtypes  = [numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # K_tot_ALL
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # betas
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # A
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # W
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # Diag
                                    ctypes.c_int,                                       # N
                                    ctypes.c_int,                                       # NA
                                    ctypes.c_int,                                       # m
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # output
                                    ctypes.c_int,                                       # i1
                                    ctypes.c_int]                                       # i2
libPyMKL.computeENERGY.restype   =  None

# COMPUTE SWA
# void    __computeSWA(double *K_tot_ALL,                double *A, double *W, double *Diag, double *S_W_A, double *S_D_A, int N, int NA, int m,                   int i1, int i2);
libPyMKL.computeSWA.argtypes     = [numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # K_tot_ALL
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # A
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # W
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # Diag
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # S_W_A
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # S_D_A
                                    ctypes.c_int,                                       # N
                                    ctypes.c_int,                                       # NA
                                    ctypes.c_int,                                       # m
                                    ctypes.c_int,                                       # i1
                                    ctypes.c_int]                                       # i2
libPyMKL.computeSWA.restype      =  None

# COMPUTE SWB
# void    __computeSWB(double *K_tot_ALL, double *betas,            double *W, double *Diag, double *S_W_B, double *S_D_B, int N,         int m,                   int i1, int i2);
libPyMKL.computeSWB.argtypes     = [numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # K_tot_ALL
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # betas
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # W
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # Diag
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # S_W_B
                                    numpy.ctypeslib.ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),   # S_D_B
                                    ctypes.c_int,                                       # N
                                    ctypes.c_int,                                       # m
                                    ctypes.c_int,                                       # i1
                                    ctypes.c_int]                                       # i2
libPyMKL.computeSWB.restype      =  None

# FUNCTION WRAPPERS
def computeENERGY(K_tot_ALL, betas, A, W, Diag, i1, i2):
    """Compute ENERGY"""
    assert type(K_tot_ALL)  == np.ndarray
    assert type(betas)      == np.ndarray
    assert type(A)          == np.ndarray
    assert type(W)          == np.ndarray
    assert type(Diag)       == np.ndarray
    assert type(i1)         == np.dtype('int')
    assert type(i2)         == np.dtype('int')

    N                       = int(K_tot_ALL.shape[0])
    NA                      = int(A.shape[1])
    m                       = int(K_tot_ALL.shape[2])
    K_tot_ALL               = np.asfortranarray(K_tot_ALL)
    betas                   = np.asfortranarray(betas)
    A                       = np.asfortranarray(A)
    W                       = np.asfortranarray(W)
    Diag                    = np.asfortranarray(Diag)
    # OUTPUT VECTOR
    output                  = np.array([0,0],dtype=ctypes.c_double,order='F')

    libPyMKL.computeENERGY(K_tot_ALL,betas,A,W,Diag,N,NA,m,output,int(i1),int(i2))

    gap     = output[0]
    constr  = output[1]

    return (gap, constr) 



def computeSWA(K_tot_ALL, A, W, Diag, i1, i2):
    """Compute SWA"""
    assert type(K_tot_ALL)  == np.ndarray
    assert type(A)          == np.ndarray
    assert type(W)          == np.ndarray
    assert type(Diag)       == np.ndarray
    assert type(i1)         == np.dtype('int')
    assert type(i2)         == np.dtype('int')

    N                       = int(K_tot_ALL.shape[0])
    NA                      = int(A.shape[1])
    m                       = int(K_tot_ALL.shape[2])
    K_tot_ALL               = np.asfortranarray(K_tot_ALL)
    W                       = np.asfortranarray(W)
    Diag                    = np.asfortranarray(Diag)
    A                       = np.asfortranarray(A)
    # OUTPUT MATRICES
    S_W_A                   = np.zeros((m,m),dtype=ctypes.c_double,order='F')
    S_D_A                   = np.zeros((m,m),dtype=ctypes.c_double,order='F')

    libPyMKL.computeSWA(K_tot_ALL,A,W,Diag,S_W_A,S_D_A,N,NA,m,int(i1),int(i2))
    
    return (np.ascontiguousarray(S_W_A), np.ascontiguousarray(S_D_A)) 



def computeSWB(K_tot_ALL, betas, W, Diag, i1, i2):
    """Compute SWB"""
    assert type(K_tot_ALL)  == np.ndarray
    assert type(betas)      == np.ndarray
    assert type(W)          == np.ndarray
    assert type(Diag)       == np.ndarray
    assert type(i1)         == np.dtype('int')
    assert type(i2)         == np.dtype('int')

    N                       = int(K_tot_ALL.shape[0])
    m                       = int(K_tot_ALL.shape[2])
    # OUTPUT MATRICES
    S_W_B                   = np.zeros((N,N),dtype=ctypes.c_double,order='F')
    S_D_B                   = np.zeros((N,N),dtype=ctypes.c_double,order='F')

    K_tot_ALL               = np.asfortranarray(K_tot_ALL)
    betas                   = np.asfortranarray(betas)
    W                       = np.asfortranarray(W)
    Diag                    = np.asfortranarray(Diag)

    libPyMKL.computeSWB(K_tot_ALL, betas, W, Diag, S_W_B, S_D_B, int(N),int(m),int(i1),int(i2))

    return (np.ascontiguousarray(S_W_B), np.ascontiguousarray(S_D_B)) 



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


