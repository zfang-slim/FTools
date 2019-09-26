import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML
# from pysit.util.util import *

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



__all__ = ['getImageFromList', 'create_animation', 'plot_model', 'plot_data', 'imagesc']

__docformat__ = "restructuredtext en"

def getImageFromList(x):
    return imageList[x]

def create_animation(vt, vi, fwave, awave, IC, Lateral, Depth, model_size, ntime, f_alpha=0.01, a_alpha=10000.0, climc=None, ksmp=50):
    plt.figure()
    fig, ax = plt.subplots(3,2,figsize=(12,12))

    ims = []
    model_sizet = [model_size[1], model_size[0]]
    DD = np.zeros(model_sizet)
    EE = np.zeros(model_sizet)
    vt = np.reshape(vt, model_size)
    vt = np.transpose(vt)
    vi = np.reshape(vi, model_size)
    vi = np.transpose(vi)
    for i in range(0,ntime,ksmp):    
        A = np.reshape(fwave[i], model_size)
        A = np.transpose(A)
        C = np.reshape(awave[i], model_size)
        C = np.transpose(C)
#         DD = A*C+DD
        DD = IC[i]
        EE = DD+C*a_alpha+A*f_alpha
        
        ax[0][0].set_title('true model')
        ax[0][0].set_xlabel('X [km]')
        ax[0][0].set_ylabel('Z [km]')
        im0=ax[0][0].imshow(vt, clim=climc[0], 
                            extent=[Lateral[0], Lateral[model_size[0]-1], Depth[model_size[1]-1], Depth[0]],
                            animated=True)
        
        ax[0][1].set_title('initial model')
        ax[0][1].set_xlabel('X [km]')
        ax[0][1].set_ylabel('Z [km]')
        im1=ax[0][1].imshow(vi, clim=climc[1], 
                            extent=[Lateral[0], Lateral[model_size[0]-1], Depth[model_size[1]-1], Depth[0]],
                            animated=True)        
        
        ax[1][0].set_title('Forward wavefield')
        ax[1][0].set_xlabel('X [km]')
        ax[1][0].set_ylabel('Z [km]')
        im2=ax[1][0].imshow(A,clim=climc[2], 
                            extent=[Lateral[0], Lateral[model_size[0]-1], Depth[model_size[1]-1], Depth[0]],
                            animated=True)
        
        ax[1][1].set_title('Adjoint wavefield')
        ax[1][1].set_xlabel('X [km]')
        ax[1][1].set_ylabel('Z [km]')
        im3=ax[1][1].imshow(C,clim=climc[3], 
                            extent=[Lateral[0], Lateral[model_size[0]-1], Depth[model_size[1]-1], Depth[0]],
                            animated=True)
        
        ax[2][0].set_title('Gradient')
        ax[2][0].set_xlabel('X [km]')
        ax[2][0].set_ylabel('Z [km]')
        im4=ax[2][0].imshow(DD,clim=climc[4], 
                            extent=[Lateral[0], Lateral[model_size[0]-1], Depth[model_size[1]-1], Depth[0]],
                            animated=True)
        
        ax[2][1].set_title('Two wavefield + gradient')
        ax[2][1].set_xlabel('X [km]')
        ax[2][1].set_ylabel('Z [km]')
        im5=ax[2][1].imshow(EE,clim=climc[5], 
                            extent=[Lateral[0], Lateral[model_size[0]-1], Depth[model_size[1]-1], Depth[0]],
                            animated=True)
        
        ims.append([im0,im1,im2,im3,im4,im5])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
    plt.close()
    
    return ani

def plot_model(fig, ax, data, Lateral, Depth, model_size, clim=None, title_str=None,cmap='jet',animated=False, colorbar_label=None, colorbarFlag=True):
    im1=ax.imshow(data,
                  clim=clim,
                  extent=[Lateral[0], Lateral[-1], Depth[-1], Depth[0]], animated=animated)
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Z [km]')
    ax.set_title(title_str)
    im1.set_cmap(cmap)
    if colorbarFlag is True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

    return im1

def imagesc(data, Lateral=None, Depth=None, clim=None, title_str=None, xlabel='None', ylabel='None',
            cmap='jet', colorbar=True, colorbar_label=None, ax=None, fig=None):
    if ax is None:
        im1 = plt.imshow(data, clim=clim, extent=[Lateral[0], Lateral[-1], Depth[-1], Depth[0]], interpolation='nearest', aspect='auto', cmap=cmap)
        if title_str is not None:
            plt.title(title_str)
        if xlabel is not None:
            plt.xlable(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if colorbar is True:
            clb=plt.colorbar()    

            if colorbar_label is not None:
                clb.ax.set_title(colorbar_label)
    else:
        im1 = ax.imshow(data, clim=clim, extent=[Lateral[0], Lateral[-1], Depth[-1], Depth[0]], interpolation='nearest', aspect='auto', cmap=cmap)
        if title_str is not None:
            ax.set_title(title_str)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if colorbar is True: 
            clb=plt.colorbar(im1, ax=ax)    
            if colorbar_label is not None:
                clb.ax.set_title(colorbar_label)


    return im1




    

    
def plot_data(ax, data, t_smp, title_str=None):
    im1=ax.plot(t_smp,data)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.set_title(title_str)
