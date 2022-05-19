import os
import os.path
import numpy as np
import scipy as sp

from numba import *

# FUNCTION WRAPPERS
@jit((float64[:,:,:],float64[:,:],float64[:,:],float64[:,:],float64[:],int64,int64), nopython=True, parallel=True)
def computeENERGY(K_tot_all: np.ndarray, betas: np.ndarray, A: np.ndarray, W: np.ndarray, Diag: np.ndarray, i1, i2):
    N,N,M = K_tot_all.shape
    NA    = A.shape[1]
    
    tmp    = np.zeros((NA))
    tmpM   = np.zeros((N,N))
    gap    = np.float64(0.0)
    constr = np.float64(0.0)

    for i in range(N):
        for r in range(N):
            tmpV = np.float64(0.0)
            for c in range(M):
                tmpV += K_tot_all[i,r,c] * betas[c,0]
            tmpM[i,r] = tmpV

    for i in range(N):
        for j in range(i+1,N):
            Wij = W[i,j]
            if Wij != 0:
                for r in range(NA):
                    tmpV = np.float64(0.0)
                    for l in range(N):
                        tmpV += A[l,r] * (tmpM[i,l] - tmpM[j,l])
                    tmp[r] = tmpV

                gapij = np.float64(0.0)
                for r in range(NA):
                    gapij += tmp[r]**2
                gap += 2.0 * gapij * Wij

        for r in range(NA):
            tmpV = np.float64(0.0)
            for l in range(N):
                tmpV += A[l,r] * tmpM[i,l]
            tmp[r] = tmpV

        constrii = np.float64(0.0)
        for r in range(NA):
            constrii += tmp[r]**2
        constr += Diag[i] * constrii
                
    return gap,constr


@jit((float64[:,:,:],float64[:,:],float64[:,:],float64[:],int64,int64), nopython=True, parallel=True)
def computeSWA(K_tot_all: np.ndarray, A: np.ndarray, W: np.ndarray, Diag: np.ndarray, i1, i2):
    N,N,M = K_tot_all.shape
    NA    = A.shape[1]
    
    tmp      = np.zeros((M,NA))
    KiA      = np.zeros((N,M,NA))
    SWA = np.zeros((M, M))
    SDA = np.zeros((M, M))

    for i in range(N):
        for c in range(M):
            for r in range(NA):
                tmpV = np.float64(0.0)
                for l in range(N):
                    tmpV += K_tot_all[i,l,c] * A[l,r]
                KiA[i,c,r] = tmpV

    for i in range(N):
        for j in range(i+1,N):
            Wij = W[i,j]
            if Wij != 0:
                for c1 in range(M):
                    for c2 in range(c1,M):
                        tmpV = np.float64(0.0)
                        for k in range(NA):
                            tmp1  = KiA[i,c1,k] - KiA[j,c1,k]
                            tmp2  = KiA[i,c2,k] - KiA[j,c2,k]
                            tmpV += tmp1 * tmp2
                        SWA[c1,c2] += 2.0 * Wij * tmpV
        
        for c in range(M):
            for r in range(NA):
                tmpV = np.float64(0.0)
                for l in range(N):
                    tmpV += K_tot_all[i,l,c] * A[l,r]
                tmp[c,r] = tmpV
        
        for c1 in range(M):
            for c2 in range(c1,M):
                tmpV = np.float64(0.0)
                for k in range(NA):
                    tmpV += tmp[c1,k] * tmp[c2,k]
                SDA[c1,c2] += Diag[i] * tmpV
                
    return np.ascontiguousarray(SWA), np.ascontiguousarray(SDA)


@jit((float64[:,:,:],float64[:,:],float64[:,:],float64[:],int64,int64), nopython=True, parallel=True)
def computeSWB(K_tot_all: np.ndarray, betas: np.ndarray, W: np.ndarray, Diag: np.ndarray, i1, i2):
    N,N,M = K_tot_all.shape
    
    tmp      = np.zeros((N,))
    SW_betas = np.zeros((N, N))
    SD_betas = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i+1,N):
            Wij = W[i,j]
            
            if Wij != 0:
                for r in range(N):
                    tmpV = np.float64(0.0)
                    for c in range(M):
                        tmpV += (K_tot_all[i,r,c] - K_tot_all[j,r,c]) * betas[c,0]
                    tmp[r] = tmpV
                
                for r in range(N):
                    for l in range(r,N):
                        SW_betas[r,l] += 2*Wij * (tmp[r] * tmp[l])
        
        for r in range(N):
            tmpV = np.float64(0.0)
            for c in range(0,M):
                tmpV += K_tot_all[i,r,c] * betas[c,0]
            tmp[r] = tmpV
            
        for r in range(N):
            for l in range(r,N):
                SD_betas[r,l] += Diag[i] * (tmp[r] * tmp[l])
                
    return np.ascontiguousarray(SW_betas), np.ascontiguousarray(SD_betas)

