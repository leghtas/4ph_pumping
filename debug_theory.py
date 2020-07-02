# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:11:49 2016

@author: leghtas
"""

import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import circuit as cc
import circuit_PCSSv7 as cPCSSv7
import scipy.linalg as sl
import numpy.linalg as nl
from scipy.misc import factorial
from functions_tools import *
from matplotlib.animation import FuncAnimation
from get_saddle_general import get_saddle

e = cc.e



pi = np.pi
plt.close('all')

c = cPCSSv7.CircuitPump2Snail(Csum=242e-15, LG=0.159e-9, LK=4.25e-9, LJ=1.86e-9) #LJ = 1.86nH
EJ = c.EJ
ECsum = c.ECsum
EL = c.ELK

phi_Delta = np.linspace(pi-pi/8, pi+pi/8, 31)
phi_Sum = np.linspace(pi-pi/15, pi+pi/15, 33)

# Get freqs and Kerrs v.s. flux

max_solutions = 1
n_modes = c.dim
print('Found %d modes...'%(n_modes))
Xi2 = np.zeros((len(phi_Delta), len(phi_Sum), max_solutions, n_modes))
f_calculation = np.zeros((len(phi_Delta), len(phi_Sum)))

eJL = EJ/EL
f0 = np.sqrt(8*EL*ECsum)/(c.hbar*2*pi)/1e9
for iD, pD in enumerate(phi_Delta):
    for iS, pS in enumerate(phi_Sum):
        _res  = c.get_freqs_only(return_components=True, max_solutions=max_solutions,
                                 sort=False,pext_sigma=pS, pext_delta=pD)
        res1, res2, Xi2s, P= _res
        Xi2[iD, iS] = Xi2s
        dpS = (pS - pi)/2
        dpD = (pD - pi)/2
        f_calculation[iD, iS] = f0*(1+2*eJL*dpS*(dpD+eJL*dpS))
    
Xi2 = np.moveaxis(Xi2, -1, 0)

fig0, ax0 = plt.subplots(1,2, figsize=(12,10))
f_th = Xi2[:,:,:,0]/1e9 # GHz
f_th_plot  = np.where(f_th[0]>4, f_th[0], np.nan)
f_th_plot  = np.where(f_th_plot<8, f_th_plot, np.nan)

get_saddle(phi_Sum, phi_Delta, f_th[0])
get_saddle(phi_Sum, phi_Delta, f_calculation)

cc.pcolor_z(ax0[0], phi_Sum, phi_Delta, f_th[0], vmin=[4.45,6.0])
cc.pcolor_z(ax0[1], phi_Sum, phi_Delta, f_calculation, vmin=[4.45,6.0])
ax0[0].set_xlabel('phi_sum (rad)')
ax0[0].set_ylabel('phi_delta (rad)')

ax0[1].set_xlabel('phi_sum (rad)')
ax0[1].set_ylabel('phi_delta (rad)')

ax0[0].set_title('SIMULATION')
ax0[1].set_title('CALCULATION')