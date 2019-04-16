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



LJ = 12000e-12
w = 3e9*2*pi
wa, Za = [8e9*2*np.pi, 50]


Cc = 10e-16 #F

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

    print('g/Delta = '+str(c.g/np.abs(wa-w)))
    phiZPFa = c.phiZPFa
    phiZPFj = c.phiZPFj
    kerr_mem = 2*EJ/24*phiZPFa**4*(c.g/np.abs(wa-w))**4*6/c.hbar/2/np.pi*1e-6
    cross_kerr = EJ/24*phiZPFa**2*phiZPFj**2*(c.g/np.abs(wa-w))**2*4/c.hbar/2/np.pi*1e-6/2
    kerr = 2*EJ/24*phiZPFj**4*6/c.hbar/2/np.pi*1e-6
    print('cross_kerr = %.4f MHz'%cross_kerr)
    print('kerr = %.3f MHz'%kerr)
    
    wa_t = c.wa_t
    w_t = c.w_t
    bogo = np.array([[wa_t, 0, c.g, -c.g], 
                     [0, -wa_t, c.g, -c.g], 
                     [c.g, -c.g, w_t, 0],
                     [c.g, -c.g, 0, -w_t]])
    e, v = np.linalg.eig(bogo)
    

    print(e/2/np.pi*1e-9)
    print(v)
    inv_v = np.linalg.inv(v)
    
#    print(c.phiZPFj_t*v[0])
    print('phib original')
    print(c.phiZPFj_t*(inv_v[0]+inv_v[3]))
    print(c.phiZPFj_t*(inv_v[1]+inv_v[2]))
    
    print('phiZPF')
    print(phiZPFa, phiZPFj)

    cross_Kerr = EJ/24*P[0,0]**2*P[0,1]**2
    
    print('\ntest freq')
    exp_omegaO2 = 2*c.EJ/c.hbar*P[0,0]**2
    print(exp_omegaO2/2/np.pi*1e-9)
    print(Xi2[0]*1e-9)
    print('\ntest_Kerr')
    exp_Kerr = c.EJ/c.hbar/24*P[0,0]**4*2*6
    print(exp_Kerr/2/np.pi*1e-6)
    print(Xi4[0]*1e-6)
    
    print('\ntest_Kerr_mem')
    exp_Kerr = c.EJ/c.hbar/24*P[0,1]**4*2*6
    print(exp_Kerr/2/np.pi*1e-6)
    print(Xi4[1]*1e-6)
    
    print('\ntest_cross_Kerr')
    exp_Kerr = c.EJ/c.hbar/24*P[0,1]**2*P[0,0]**2*4*6
    print(exp_Kerr/2/np.pi*1e-6)
    print(Xip[0]*1e-6)

    

    print('\nf_cav = %.5f GHz'%(Xi2[1]/1e9))
    print('Kerr_cav = %.3f MHz'%(Xi4[1]/1e6))
    
    print('\ncross_Kerr = %.3f MHz'%(Xip[0]/1e6))
    
    print('\nf_trm = %.3f GHz'%(Xi2[0]/1e9))
    print('Kerr_trm = %.3f MHz\n'%(Xi4[0]/1e6))
    
    print('ratio_kerr = %.3f'%(Xi4[0]/1e6/kerr))
    print('ratio_cross = %.3f'%(Xip[0]/1e6/cross_kerr))
    print('ratio_mem = %.3f'%(Xi4[1]/1e6/kerr_mem))