# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:54:12 2018

@author: Guillermo Jimenez-Perez
         Mariana Nogueira
"""

import numpy as np
import picos as pic
import warnings
import tqdm
from scipy.linalg import eig
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from cvxopt import matrix
try:
    from joblib import Parallel
    from joblib import delayed
    from joblib.parallel import effective_n_jobs
except ImportError:
    pass

import lib


# class MKL():

#     def __init__(self, X: Union[List[np.ndarray], Dict[str, np.ndarray]], maxiter: int = 25, sig_knn: int = None, 
#                        sparse_knn: int = None, eps: float = 1e-6, precision: str = "low", verbose: bool = True):
#         """Unsupervised Multiple Kernel Learning formulation.
#         Inputs:
#         * X:            List of M numpy arrays, each consisting of a NxD stack of the samples under those features.
#         * maxiter:      Maximum allowed number of iterations.
#         * sig_knn:      Number of neighbours for the kernel. Defaults to the square root of the number of samples.
#         * sparse_knn:   Number of neighbours for the sparsing of the global affinity matrix. Defaults to the square root of the number of samples.
#         * eps:        Machine epsilon.
#         * precision:    Precision of the convex optimization algorithm
#         """

#         # Store inputs
#         if isinstance(X, list):
#             names = list(range(len(X)))
#         elif isinstance(X, dict):
#             names = [k for k in X]
#             X = [X[k] for k in names]
#         else:
#             raise ValueError(f"Input X can only be a list of np.ndarray or a dict. Got {type(X)}")
#         self.X          = X 
#         self.names      = names 
#         self.sparse_knn = sparse_knn
#         self.sig_knn    = sig_knn
#         self.iter       = 0
#         self.maxiter    = maxiter
#         self.precision  = precision
#         self.M          = len(X)
#         self.N          = X[0].shape[0]
#         if np.all(np.diff([X[i].shape[0] for i in range(self.M)]) == 0):
#             if np.all(np.diff([X[i].shape[1] for i in range(self.M)]) == 0):
#                 warnings.warn("Observations provided in the second dimension of the arrays of X. Transposing, but be aware of this: might be an error!")
#                 self.X  = [x.T for x in self.X]
#                 self.N  = X[0].shape[0]
#             else:
#                 raise ValueError("First dimension of np.ndarrays in X must be the number of observations")
#         else:
#             raise ValueError("First dimension of np.ndarrays in X must be the number of observations")
#         if self.M >= 65:
#             warnings.warn("BEWARE! The algorithm is not guaranteed to converge with 60+ input features.")

#         # Preprocess inputs
#         self.sparse_knn = sparse_knn if (sparse_knn is not None) else math.floor(np.sqrt(self.N))
#         self.sig_knn    = sig_knn    if (sig_knn    is not None) else math.floor(np.sqrt(self.N))

#         # Stack kernels for MKL
#         K, var              = __kernel_stack(X, self.M, self.N, self.sig_knn)
#         sparse_W, D         = __get_sparse_W_and_D(K, self.M, var, self.sparse_knn)

#         # Break symmetry
#         preconditioner      = 0.00*np.random.rand(self.M, 1) + np.ones((self.M, 1))

#         self.K              = K
#         self.sparse_W       = sparse_W
#         self.D              = D
#         self.var            = var
#         self.betas          = preconditioner/np.sum(preconditioner)
#         self.eps            = eps
#         self.fitted         = False
#         self.energy            = np.array([])
#         self.constr         = np.array([])
#         self.verbose        = verbose

#         if self.precision.lower() not in ["low","medium","high","best"]:
#             raise
#         else:
#             # See http://cvxr.com/cvx/doc/solver.html#controlling-precision
#             if self.precision.lower() == "low":
#                 self.__precision_eps = np.finfo(np.double).eps**(3./8)
#             elif self.precision.lower() == "medium":
#                 self.__precision_eps = np.finfo(np.double).eps**(1./2)
#             elif self.precision.lower() == "high":
#                 self.__precision_eps = np.finfo(np.double).eps**(3./4)
#             elif self.precision.lower() == "best":
#                 self.__precision_eps = np.finfo(np.double).eps

class MKL():
    def __init__(self, K: np.ndarray, W: np.ndarray, D: np.ndarray, maxiter: int = 25, 
                 eps: float = 1e-6, verbose: bool = True, solver: str = "cvxopt"):
        """Unsupervised Multiple Kernel Learning formulation.
        Inputs:
        * X:            List of M numpy arrays, each consisting of a NxD stack of the samples under those features.
        * maxiter:      Maximum allowed number of iterations.
        * eps:          Machine epsilon.
        * precision:    Precision of the convex optimization algorithm
        """

        # Store inputs
        self.K          = K.astype("float64")
        self.W          = W.astype("float64")
        self.D          = D.astype("float64")
        self.iter       = 0
        self.maxiter    = maxiter
        self.solver     = solver
        self.M          = self.K.shape[0]
        self.N          = self.K.shape[1]
        self.eps        = eps
        self.verbose    = verbose
        if self.M >= 65:
            warnings.warn("BEWARE! The algorithm is not guaranteed to converge with 60+ input features.")

        # Break symmetry
        preconditioner  = 0.00*np.random.rand(self.M, 1) + np.ones((self.M, 1))
        self.betas      = preconditioner/np.sum(preconditioner)
        self.fitted     = False
        self.energy     = []
        self.constr     = []
        self.gap        = np.nan
        self.tolerance  = np.inf


    def __compute_SWB_caller(self):  # solve for A (generalized eigenvalue problem)
        # compute sides of GEP

        try:
            numWorkers      = effective_n_jobs()
            values          = lib.create_ivalues(numWorkers,self.N)
            start_values    = values[0:-1]
            end_values      = values[1:]

            SW_betas        = np.zeros((self.N, self.N))
            SD_betas        = np.zeros((self.N, self.N))

            ParallelResult  = Parallel(n_jobs=numWorkers, prefer="threads")(delayed(lib.computeSWB)(np.moveaxis(self.K,0,-1),self.betas,self.W,np.diag(self.D),start_values[i],end_values[i]) for i in range(len(start_values)))    

            for (SW_betastmp,SD_betastmp) in ParallelResult:
                SW_betas    = SW_betas + SW_betastmp
                SD_betas    = SD_betas + SD_betastmp
        except NameError:
            (SW_betas,SD_betas) = lib.computeSWB(np.moveaxis(self.K,0,-1),self.betas,self.W,np.diag(self.D),0,self.N)
        except KeyboardInterrupt:
            raise

        SW_betas            = SW_betas + np.triu(SW_betas,1).T
        SD_betas            = SD_betas + np.triu(SD_betas,1).T

        ## TRANSFORM THE MATRICES TO POSITIVE DEFINITE ##
        SW_betas            = np.real(lib.to_PDM(SW_betas, self.eps)) + self.eps*np.eye(self.N)
        SD_betas            = np.real(lib.to_PDM(SD_betas, self.eps)) + self.eps*np.eye(self.N)

        # Avoid numerical errors
        SW_betas            = 0.5*(SW_betas + SW_betas.T)
        SD_betas            = 0.5*(SD_betas + SD_betas.T)

        if np.allclose(np.matrix(SW_betas),np.matrix(SW_betas).H):
            # If is Hermitian, perform scipy's eigh (WAY faster)
            eigvals, A      = eigh(SW_betas, b=SD_betas)
        else:
            # Else, compute scipy's eig
            eigvals, A      = eig(SW_betas,  b=SD_betas)

        A                   = np.real(A)
        eigvals             = np.real(eigvals)
        A                   = A[:,(eigvals > self.eps).ravel()]
        eigvals             = eigvals[(eigvals > self.eps).ravel()]

        # Normalization to comply with original MATLAB code
        A                   = np.divide(A,np.linalg.norm(A,axis=0))

        self.A              = A
        self.SW_betas       = SW_betas
        self.SD_betas       = SD_betas

    def __compute_SWA_caller(self):  # semidefinite optimization problem
        # Use C compiled code
        try:
            numWorkers      = effective_n_jobs()
            values          = lib.create_ivalues(numWorkers,self.N)
            start_values    = values[0:-1]
            end_values      = values[1:]

            SW_A            = np.zeros((self.M, self.M))
            SD_A            = np.zeros((self.M, self.M))

            ParallelResult  = Parallel(n_jobs=numWorkers, prefer="threads")(delayed(lib.computeSWA)(np.moveaxis(self.K,0,-1),self.A,self.W,np.diag(self.D),start_values[i],end_values[i]) for i in range(len(start_values)))    

            for (SW_Atmp,SD_Atmp) in ParallelResult:
                SW_A        = SW_A + SW_Atmp
                SD_A        = SD_A + SD_Atmp
        except NameError:
            (SW_A,SD_A)     = lib.computeSWA(np.moveaxis(self.K,0,-1),self.A,self.W,np.diag(self.D),0,self.N)
        except KeyboardInterrupt:
            raise

        # Solve solution throwing double the result in C compiled code
        SW_A                = (SW_A + np.triu(SW_A,1).T)
        SD_A                = (SD_A + np.triu(SD_A,1).T)
        
        SW_A                = 0.5 * (SW_A + SW_A.T)
        SD_A                = 0.5 * (SD_A + SD_A.T)

        self.SW_A           = SW_A
        self.SD_A           = SD_A

    def __convex_optimization(self):
        sdp                 = pic.Problem()
        B                   = sdp.add_variable('B',(self.M,self.M),vtype='symmetric')
        betas               = sdp.add_variable('betas',(self.M,1))
        sdp.set_objective('min','I'|matrix(self.SW_A)*B)
        sdp.add_constraint('I'|matrix(self.SD_A)*B >= 0)
        sdp.add_constraint(betas >= 0)
        # sdp.add_constraint(pic.sum([betas[i] for i in range(self.M)],'i','0...'+str(self.M)) == 1)
        sdp.add_constraint(((1 & betas.T) // (betas & B))>>0 )

        sdp.solve(solver=self.solver, solve_via_dual=False, verbose=False)

        betas               = np.array(betas.value)
        betas               = betas/np.sum(betas)

        self.betas          = betas

    def __compute_ENERGY_caller(self):
        try:
            numWorkers      = effective_n_jobs()
            values          = lib.create_ivalues(numWorkers,self.N)
            start_values    = values[0:-1]
            end_values      = values[1:]

            gap             = 0
            constr          = 0

            ParallelResult = Parallel(n_jobs=numWorkers, prefer="threads")(delayed(lib.computeENERGY)(np.moveaxis(self.K,0,-1), self.betas, self.A, self.W, np.diag(self.D), start_values[i],end_values[i]) for i in range(len(start_values)))

            for (gaptmp,constrtmp) in ParallelResult:
                gap         = gap    + gaptmp
                constr      = constr + constrtmp

            gap             = gap
            constr          = constr
        except NameError:
            (gap, constr)   = lib.computeENERGY(np.moveaxis(self.K,0,-1), self.betas, self.A, self.W, np.diag(self.D), 0, self.N)
        except KeyboardInterrupt:
            raise

        # Store energy
        self.energy.append(gap)
        self.constr.append(constr)


    def fit(self):
        if not self.fitted:
            iterator = tqdm.tqdm(np.arange(1,self.maxiter+1), 
                                 desc="Iteration {:>3d}/{:>3d}, Energy {:10.6f} (> tolerance {:10.6f})".format(self.iter, self.maxiter, self.gap, self.tolerance))

            for self.iter in iterator:
                # Optimize model
                self.__compute_SWB_caller()
                self.__compute_SWA_caller()
                self.__convex_optimization()
                self.__compute_ENERGY_caller()


                # Get gap
                self.gap = np.diff(self.energy[::-1])[0] if len(self.energy) > 1 else np.nan

                # Set tolerance
                if self.iter == 2:
                    self.tolerance = np.abs(self.energy[0] - self.energy[1])*0.01;
                
                # Break if convergence
                if np.abs(self.gap) <= self.tolerance:
                    iterator.set_description("Iteration {:>3d}/{:>3d}, Energy {:10.6f} (< tolerance {:10.6f}; BREAK!)".format(self.iter, self.maxiter, self.gap, self.tolerance),refresh=True)
                    break
                # Update iterator
                iterator.set_description("Iteration {:>3d}/{:>3d}, Energy {:10.6f} (> tolerance {:10.6f})".format(self.iter, self.maxiter, self.gap, self.tolerance),refresh=False)

            if self.iter == self.maxiter:
                print("Converged!")
        else:
            print("Model is already trained")

        
    def fit_transform(self):
        if not self.fitted:
            self.fit()
            return self.transform()
        else:
            return self.transform()

    def transform(self):
        try:
            try:
                return self.proj
            except AttributeError:
                if self.A.size != 1:
                    dim     = self.A.shape[1]
                    self.proj = np.zeros((self.N, dim))
                    for i in range(self.N):
                        Ki  = self.K[:, i, :].transpose()
                        self.proj[i] = self.A.transpose().dot(Ki).dot(self.betas).transpose()

                    return self.proj
        except AttributeError:
            print(" * Unsupervised MKL has not been trained. Cannot transform the data")









