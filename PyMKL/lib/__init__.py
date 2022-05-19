import os
PyMKL.lib.cpp = None
import PyMKL.lib.numba
try:
    import PyMKL.lib.cpp
    from PyMKL.lib.cpp import computeENERGY
    from PyMKL.lib.cpp import computeSWA
    from PyMKL.lib.cpp import computeSWB
    flagCpp = True
except:
    from PyMKL.lib.numba import computeENERGY
    from PyMKL.lib.numba import computeSWA
    from PyMKL.lib.numba import computeSWB
    flagCpp = False
