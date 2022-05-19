import os
from .__ops_numba import computeENERGY
from .__ops_numba import computeSWA
from .__ops_numba import computeSWB
from .__ops       import to_PDM
from .__ops       import create_ivalues
flagCpp = False
if os.path.exists(os.path.join(os.path.dirname(__file__),'libPyMKL.so')):
    try:
        from .__ops_cpp import computeENERGY
        from .__ops_cpp import computeSWA
        from .__ops_cpp import computeSWB
        flagCpp = True
    except:
        pass
