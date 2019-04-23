# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:11:49 2016

@author: leghtas
"""

import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import circuit
import circuit_trm_cav as ctc
import scipy.linalg as sl
import numpy.linalg as nl
from scipy.misc import factorial
import matplotlib
from matplotlib.animation import FuncAnimation
import sys

e = circuit.e

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)
    plt.show()

pi = np.pi
plt.close('all')



LJ = 6000e-12
w = 4e9*2*pi
wa, Za = [8e9*2*np.pi, 50]


Cc = 100e-16 #F

_, _, EJ = circuit.get_E_from_w(1, 1, LJ) #dLJ should be 0 to compute non-linearities !!!!!

#EJ, LJ, I0 = circuit.convert_EJ_LJ_I0(I0=I0)


c = ctc.CircuitTrmCav(w, EJ, wa, Za, Cc)

min_phi = 0*2*pi
max_phi = 1*2*pi
Npts = 1
phiVec = np.linspace(min_phi, max_phi, Npts)
ng_sweep = np.linspace(-1, 1, 21)

min_Cca = 1e-15
max_Cca = 100000e-15
Npts = 101
CcaVec = np.linspace(min_Cca, max_Cca, Npts)
ECcaVec = e**2/2/CcaVec



# Get freqs and Kerrs v.s. flux
# Get freqs and Kerrs v.s. flux
if 1==1:
    n_modes = c.dim
    max_solutions = 1
    particulars = [(0,0,1,1)] # term in front of (a+a^+)**2(b+b^+)**2
    factors_particulars = np.array([4]) # term in front of (a+a^+)**2(b+b^+)**2
    
    _res  = c.get_freqs_kerrs(particulars=particulars, return_components=True, max_solutions=max_solutions)
    res1, res2, Xi2s, Xi3s, Xi4s, Xips, Ps = _res
    # [0] cause only one solution
    c.print_P(Ps[0]) 
    P = Ps[0]
    Xi2 = Xi2s[0] 
    Xi3 = Xi3s[0]
    Xi4 = 2*Xi4s[0]
    Xip = Xips[0]*factors_particulars
    

    print('\nf_cav = %.5f GHz'%(Xi2[1]/1e9))
    print('Kerr_cav = %.3f MHz'%(Xi4[1]/1e6))
    
    print('\ncross_Kerr = %.3f MHz'%(Xip[0]/1e6))
    
    print('\nf_trm = %.3f GHz'%(Xi2[0]/1e9))
    print('Kerr_trm = %.3f MHz\n'%(Xi4[0]/1e6))