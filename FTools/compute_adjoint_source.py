import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import copy as copy
import math
import os
import scipy.io as sio
from scipy.signal import hilbert
from shutil import copy2

import sys


__all__ = ['Envelopt_fun','Correlate_fun']

__docformat__ = "restructuredtext en"

def Envelopt_fun(dobs, dpred, p=2.0):
    dpred_Hilbert = hilbert(dpred, axis=0).imag
    dobs_Hilbert = hilbert(dobs, axis=0).imag

    dpred_envelope = dpred**2.0 + dpred_Hilbert**2.0
    dobs_envelope = dobs**2.0 + dobs_Hilbert**2.0

    resid = dpred_envelope**(p/2.0) - dobs_envelope**(p/2.0)
    
    denvelope_ddata = p * dpred_envelope**(p/2.0 - 1.0) * dpred
    adjoint_src = denvelope_ddata * resid

    denvelope_ddataH = p * dpred_envelope**(p/2.0 - 1.0) * dpred_Hilbert 
    adjoint_src += (-hilbert(denvelope_ddataH * resid, axis=0)).imag
    
    return -resid, -adjoint_src

def Correlate_fun(dobs, dpred, dt):
    shape_dobs = np.shape(dobs)
    n_correlate_data = shape_dobs[0]
    resid = np.zeros([n_correlate_data, shape_dobs[1]])
    adjoint_src = np.zeros([shape_dobs[0], shape_dobs[1]])
    W  = np.zeros(n_correlate_data)
    if np.mod(n_correlate_data, 2) == 0:
        W[0:n_correlate_data//2] = np.linspace(-dt, -dt*(shape_dobs[0])/2.0, n_correlate_data/2)
        W[n_correlate_data//2:n_correlate_data] = np.flipud(W[0:n_correlate_data//2])
    else:
        W[0:(n_correlate_data+1)//2] = np.linspace(0.0, -dt*(shape_dobs[0]-1)/2.0, (n_correlate_data+1)/2)
        W[(n_correlate_data-1)//2:n_correlate_data] = np.flipud(W[0:(n_correlate_data+1)//2])
        
    for i in range(0, shape_dobs[1]):
        correlate_data = correlate_fun(dobs[:,i], dpred[:,i])
        Wf = W * correlate_data
        correlate_data_norm2 = np.dot(correlate_data, correlate_data)
        Wf_norm2 = np.dot(Wf, Wf)
        resid[:, i] = Wf / np.sqrt(correlate_data_norm2)
            
        adjoint_src[:,i] = (2.0*correlate_data_norm2*correlate_fun(dobs[:,i], W*Wf, mode='adj') - 2.0*Wf_norm2*correlate_fun(dobs[:,i], correlate_data, mode='adj')) / correlate_data_norm2**2.0
    
    return resid, -1*adjoint_src

