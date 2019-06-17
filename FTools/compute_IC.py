import time
import copy

import numpy as np


__all__ = ['compute_IC']

__docformat__ = "restructuredtext en"

def compute_IC(fwave, awave, model_size):
    IC = []
    ic = np.zeros([model_size[1],model_size[0]])
    for i in range(0,len(fwave)):
        A = np.reshape(fwave[i], model_size)
        A = np.transpose(A)
        C = np.reshape(awave[i], model_size)
        C = np.transpose(C)
        ic = A * C + ic
        IC.append(ic)
        
    return IC
