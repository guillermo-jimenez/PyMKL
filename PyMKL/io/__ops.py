import os
import os.path
import struct
import numpy as np
import scipy as sp
import scipy.linalg

def read_mkl_binary(path,dtype="float32"):
    # Read binary file
    with open(path, "rb") as f:
        binary = f.read()
    
    # Retrieve shape
    shape = struct.unpack("ii",binary[0:8])
    
    # Get in numpy
    return np.reshape(np.frombuffer(binary[8:],dtype=dtype),shape,order="F")

def read_mkl_kernels(path,dtype="float32"):
    # Read binary file
    with open(path, "rb") as f:
        binary = f.read()
    
    # Retrieve shape
    N_kernels,N_W = struct.unpack("ii",binary[0:8])
    
    # Get in numpy
    decoded_binary = np.frombuffer(binary[8:],dtype=dtype)
    kernels = np.reshape(decoded_binary,(N_kernels,N_W,N_W),order="F")
    kernels = np.swapaxes(kernels,1,2)

    return kernels

def write_mkl_binary(matrix,path,dtype="float32"):
    if matrix.ndim != 2:
        raise ValueError(f"Incorrect dimensions: inputted {matrix.ndim}, expected 2")
        
    matrix = np.asfortranarray(matrix.copy())
    matrix = matrix.astype(dtype)
    
    # Select struct data
    if   matrix.dtype == "float64": format = "d"
    elif matrix.dtype == "float32": format = "f"
    else: raise NotImplementedError("Not yet implemented")
    
    # Write binary file
    with open(path, "wb") as f:
        f.write(struct.pack("ii",*matrix.shape))
        f.write(struct.pack(format*matrix.size,*matrix.flatten().tolist()))
    
def write_mkl_kernels(kernels,path,dtype="float32"):
    if kernels.ndim != 3:
        raise ValueError(f"Incorrect dimensions: inputted {kernels.ndim}, expected 3")
        
    kernels = np.asfortranarray(kernels.copy())
    kernels = kernels.astype(dtype)
    
    # Select struct data
    if   kernels.dtype == "float64": format = "d"
    elif kernels.dtype == "float32": format = "f"
    else: raise NotImplementedError("Not yet implemented")
    
    # Write binary file
    with open(path, "wb") as f:
        f.write(struct.pack("ii",*kernels.shape[:2]))
        f.write(struct.pack(format*kernels.size, *np.swapaxes(np.swapaxes(kernels,0,1),1,2).flatten().tolist()))


def read_outputs(base_path):
    # Recover path information
    root, fname = os.path.split(base_path)
    fname, ext  = os.path.splitext(fname)
    
    # Get path of A's
    path_A = os.path.join(base_path,"A")
    path_betas = os.path.join(base_path,"betas")
    
    # Read A matrix and normalize eigenvectors
    with open(path_A, "rb") as f:
        binary = f.read()
    
    # Retrieve shape
    shape = struct.unpack("ii",binary[0:8])[::-1] # Inverse shape
    A = np.reshape(np.frombuffer(binary[8:],dtype='float64'),shape,order="F").T
    A = A/sp.linalg.norm(A,axis=0,keepdims=True)
    
    # Read betas
    betas = read_mkl_binary(path_betas,dtype='float64')
    
    return A, betas

    