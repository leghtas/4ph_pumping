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
import matplotlib
from matplotlib.animation import FuncAnimation
import sys




e = cc.e

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

I2 = np.load('I2_1.npy')
I1 = np.load('I1_1.npy')
freq = np.load('freq_1.npy')

freq = np.where(freq>4, freq, np.nan)
freq = np.where(freq<8, freq, np.nan)

def connex_data(array, tol):
    shape = array.shape
    ret_array = np.ones(shape)*np.nan
    for jj in range(shape[0]):
        for ii in range(shape[1]):
            if array[jj, ii] is not np.nan:
                at_least = 0
                at_max = 0
                if ii>0:
                    at_max += 1
                    if not np.isnan(array[jj, ii-1]):
                        at_least+= 1
                if jj>0:
                    at_max += 1
                    if not np.isnan(array[jj-1, ii]) and jj>0:
                        at_least+= 1
                if ii<shape[1]-1:
                    at_max += 1
                    if not np.isnan(array[jj, ii+1]):
                        at_least+= 1
                if jj<shape[0]-1:
                    at_max += 1
                    if not np.isnan(array[jj+1, ii]):
                        at_least+= 1
                if (at_max-at_least)<4-tol:
                    ret_array[jj, ii] = array[jj, ii]
    return ret_array

    

fig, ax = plt.subplots(2,2, figsize= (8, 6))
cc.pcolor_z(ax[0, 0], I1, I2, freq)
filtered_freq = connex_data(freq, 2)
filtered_freq = connex_data(filtered_freq, 1)
cc.pcolor_z(ax[1, 0], I1, I2, filtered_freq)

freq = filtered_freq


wb, Zb = [4e9*2*np.pi, 50]
LJ = 3e-9 #Henri
L = 6e-9

c = cPCSSv7.CircuitPump2Snail(wb, Zb, LJ, L,)

max_freq_displayed = 15# GHz

min_phi = -1*2*pi
max_phi = 2*2*pi
Npts = 51
phi_Delta = np.linspace(min_phi, max_phi, Npts)
phi_Sum = np.linspace(min_phi, max_phi, Npts)
ng_sweep = np.linspace(-1, 1, 21)

# Get freqs and Kerrs v.s. flux
if 1==1:
    n_modes = c.dim
    print('Found %d modes...'%(n_modes))
    max_solutions = 1
    particulars = [(0,0)]#(0,0,0,1), (0,0,1,1)]
    factors_particulars = np.array([])
        
    Xi2 = np.zeros((len(phi_Delta), len(phi_Sum), max_solutions, n_modes))
    Xi3 = np.zeros((len(phi_Delta), len(phi_Sum), max_solutions, n_modes))
    Xi4 = np.zeros((len(phi_Delta), len(phi_Sum), max_solutions, n_modes))
    
    P0s = np.zeros((len(phi_Delta), len(phi_Sum)))
    
    n_particulars = len(particulars)
    Xip = np.zeros((len(phi_Delta), len(phi_Sum), max_solutions, n_particulars))

    comp = np.zeros((len(phi_Sum), max_solutions, n_modes, n_modes))
    for ll, yy in enumerate(phi_Delta):
        for kk, xx in enumerate(phi_Sum):
            _res  = c.get_freqs_kerrs(particulars=particulars, return_components=True, max_solutions=max_solutions, sort=False, pext_1=(xx+yy)/2, pext_2=(xx-yy)/2)
            res1, res2, Xi2s, Xi3s, Xi4s, Xips, P= _res
            P0s[ll, kk] = res1[0, 0]
            Xi2[ll, kk] = Xi2s
#            Xi3[kk] = Xi3s
            Xi4[ll,kk] = 2*Xi4s
            Xip[ll, kk] = Xips
#            comp[kk] = np.moveaxis(P, 1, -1)
        
    Xi2 = np.moveaxis(Xi2, -1, 0)
    Xi3 = np.moveaxis(Xi3, -1, 0)
    Xi4 = np.moveaxis(Xi4, -1, 0)
    Xip = np.moveaxis(Xip*factors_particulars, -1, 0)

#2D freq_map
if 1==1:
    plt.close('all')
    _phi_Sum, _phi_Delta = cc.to_pcolor(phi_Sum, phi_Delta)
#    Xi2[3] = np.where(Xi2[3]<10e9, Xi2[3], Xi2[1])
    colors = ['b', 'r', 'y', 'g', 'o']
    fig0, ax0 = plt.subplots(2,figsize=(12,6))
#    figkerr, axkerr = plt.subplots(3,figsize=(12,6))
#    figpumping, axpumping = plt.subplots(2, figsize=(12,6))
    
    
    f = Xi2[:,:,:,0]/1e9 #MHz
    kerr = Xi4[:,:,:,0]/1e6 #GHz
#    pumping = Xip[0,:,:,0]/1e6 #MHz
#    crosskerr = Xip[1,:,:,0]/1e6 #MHz
#    for ii in range(1):
    cc.pcolor_z(ax0[0], _phi_Sum/2/pi, _phi_Delta/2/pi, f[0])#, '.', label= 'f'+str(ii), color = colors[ii])
    ax0[0].set_title('f%d'%(0))
    
    cc.pcolor_z(ax0[1], _phi_Sum/2/pi, _phi_Delta/2/pi, P0s)#, '.', label= 'f'+str(ii), color = colors[ii])
    ax0[1].set_title('Pos min')
            #    ax0.plot(phiVec/2/pi, Xi2[2,:]/1e9, '.', label= 'f2')
#    ax0.plot(phiVec/2/pi, Xi2[3,:]/1e9, '.', label= 'f3')
#    ax0.legend()
#    ax0.set_ylabel('GHz')
#    ax0.set_ylim(0,max_freq_displayed)


# PLOT
if 1==0:
    colors = ['b', 'r', 'y', 'g', 'o']
    fig0, ax0 = plt.subplots(1, 2, figsize=(6,6))
    for ii, f in enumerate(Xi2):
        ax0.plot(phiVec/2/pi, f/1e9, '.', label= 'f'+str(ii), color = colors[ii])
#    ax0.plot(phiVec/2/pi, Xi2[2,:]/1e9, '.', label= 'f2')
#    ax0.plot(phiVec/2/pi, Xi2[3,:]/1e9, '.', label= 'f3')
    ax0.legend()
    ax0.set_ylabel('GHz')
    ax0.set_ylim(0,max_freq_displayed)
    index = np.argmin(np.abs(Xi2[1,:]-Xi2[0,:]-1e9))
    
    fig, ax = plt.subplots(n_modes, 3, figsize=(16,8), sharex=True)
    display_factor = 2
    for ii in range(n_modes):
        ax[ii, 0].plot(phiVec/2/pi, Xi2[ii]/1e9, '.', label= 'f'+str(ii))
        ax[ii, 0].legend()
        ax[ii, 0].set_ylabel('GHz')
        
        ax[ii, 1].plot(phiVec/2/pi, Xi3[ii]/1e6)
        ax[ii, 1].set_ylabel('MHz')
        mean = np.nanmean(Xi3[ii]/1e6)
        std = np.nanmean(Xi3[ii]/1e6)
        ax[ii, 1].set_ylim(mean-display_factor*std, mean+display_factor*std)
        
        ax[ii, 2].plot(phiVec/2/pi, Xi4[ii]/1e6)   
        ax[ii, 2].set_ylabel('MHz')
        mean = np.nanmean(Xi4[ii]/1e6)
        std = np.nanmean(Xi4[ii]/1e6)
        ax[ii, 2].set_ylim(mean-display_factor*std, mean+display_factor*std)
        
    ax[0,0].set_title('$a^{+}a$')
    ax[0,1].set_title('$a^{+}a^2$')
    ax[0,2].set_title('$a^{+2}a^2/2$')
    
#    
#
#    dphiVec = (phiVec[1:]+phiVec[:-1])/2
#    figp, axp = plt.subplots(n_particulars,2, figsize=(16,8), sharex=True)
#    display_factor = 2
#    for ii in range(n_particulars):
#        axp[ii, 0].plot(phiVec/2/pi, Xip[ii]/1e6, '.', label= str(particulars[ii]))
#        axp[ii, 0].legend()
#        axp[ii, 0].set_ylabel('MHz')
#        mean = np.nanmean(Xip[ii]/1e6)
#        std = np.nanmean(Xip[ii]/1e6)
#        axp[ii, 0].set_ylim(mean-display_factor*std, mean+display_factor*std)
#        
#        dXip=np.diff(Xip[ii], axis=0)/(phiVec[1]-phiVec[0])*pi/10
#        axp[ii, 1].plot(dphiVec/2/pi, dXip/1e6, '.', label= str(particulars[ii]))
#        axp[ii, 1].legend()
#        axp[ii, 1].set_ylabel('MHz')
#        mean = np.nanmean(dXip/1e6)
#        std = np.nanmean(dXip/1e6)
#        axp[ii, 1].set_ylim(mean-display_factor*std, mean+display_factor*std)
#        
#
#    
##    print('\nf0 = %.3f GHz\n'%(Xi2[0,:][0]/1e9)+'f1 = %.3f GHz'%(Xi2[1,:][0]/1e9))
##    print('k0 = %.3f MHz\n'%(Xi4[0,:][0]/1e6)+'k1 = %.3f MHz\n'%(Xi4[1,:][0]/1e6)+'k01 = %.3f MHz'%(Xi_a2s2[0]/1e6))
##    
##    print('\nphi_ext_opt = %.3f'%(phiVec[index]/2/pi))
##    print('f0 = %.3f GHz\n'%(Xi2[0,index]/1e9)+'f1 = %.3f GHz'%(Xi2[1,index]/1e9))
##    print('k0 = %.3f MHz\n'%(Xi4[0,index]/1e6)+'k1 = %.3f MHz\n'%(Xi4[1,index]/1e6)+'k01 = %.3f MHz'%(Xi_a2s2[index]/1e6))
#    
#    fig, ax = plt.subplots(2, 4, figsize=(16,8))
#    ax[0,0].plot(phiVec/2/pi, Xi2[0]/1e9, '.', label= 'f0')
#    ax[1,0].plot(phiVec/2/pi, Xi2[1]/1e9, '.', label= 'f1')
#    ax[0,0].set_title('freq')
#
#    ax[0,0].legend()
#    ax[1,0].legend()
#
#    ax[0,1].plot(phiVec/2/pi, Xi3[0]/1e6)    
#    ax[1,1].plot(phiVec/2/pi, Xi3[1]/1e6)
#    ax[0,1].set_title('c3')
#    
#    ax[0,2].plot(phiVec/2/pi, Xi4[0]/1e6)    
#    ax[1,2].plot(phiVec/2/pi, Xi4[1]/1e6)
#    ax[0,2].set_title('c4')
#
#    fig2, ax2 = plt.subplots(2,2, figsize=(16,8))
#    ax2[0,0].plot(phiVec/2/pi, Xip[0,:,0]/1e6, '.', label= '$a^2b^{+}$')
#    
#    dXi_a2s=np.diff(Xip[0,:,0])/(phiVec[1]-phiVec[0])*pi/10
#    dphiVec = (phiVec[1:]+phiVec[:-1])/2
#    
#    ax2[0,0].plot(dphiVec/2/pi, dXi_a2s/1e6, '.', label= r'$\frac{\pi}{10}*da^2b^{+}$')
#    ax2[0,0].legend()
#    ax2[0,0].set_ylabel('MHz')
#    
#    Xi4_0 = (Xi4[0,1:,0]+Xi4[0,:-1,0])/2
#    ax2[1,0].plot(dphiVec/2/pi, Xi4_0/1e6, '.', label= '$a^2a^{+2}$')
#    ax2[1,0].plot([min(dphiVec/2/pi), max(dphiVec/2/pi)], [0,0])
#    ax2[1,0].legend()
#    ax2[1,0].set_ylabel('MHz')
#
#    ax2[1,1].plot(phiVec/2/pi, Xip[1,:,0]/1e6, '.', label= r'$a^{+}ac^{+}c$')
##    ax2[1,1].plot(phiVec/2/pi, Xi_bc/1e6, '.', label= r'$b^{+}bc^{+}c$')
#    ax2[1,1].legend()
##    print('a2b = %.3f MHz\n'%(dXi_a2s[index]/1e6))
#    
#    ax2[0,1].plot(dphiVec/2/pi, dXi_a2s/Xi4_0, '.', label= '$da^2b^{+}/a^2a^{+2}$')
##    ax[3,1].plot(phiVec/2/pi, Xi4[1,:]/1e6, label= '$b^2b^{+2}$')
#    ax2[0,1].legend()
#    
#    index = np.argmin(np.abs(Xi4[0,:500,0]))
#    print('index_0c4')
#    print(index)
#    index = np.argmin(np.abs(Xip[1,:500,0]))
#    print(index)
#    print('da^+a^2 = %.3f MHz'%(dXi_a2s[index]*1e-6))
#    print('f_snail = %.3f GHz'%(Xi2[1,index,0]*1e-9))
#    print('\n')
#    index_max = np.argmax(dXi_a2s[:400])
#    print('index_max_c3')
#    print(index_max)
#    print('a^2c^+ = %.3f MHz'%(dXi_a2s[index_max]*1e-6))
#    print('a^+2a^2 = %.3f MHz'%(Xi4[0,index_max,0]*1e-6))
#    print('a^+ac^+c = %.3f MHz'%(Xip[1,index_max,0]*1e-6))
#    print('f_snail = %.3f GHz'%(Xi2[1,index_max,0]*1e-9))
#    
#    print('\n')
#    index_max = np.argmin(np.abs(Xi2[1,:500,0]-Xi2[0,:500,0]+2e9))
#    print('index_2GHz')
#    print(index_max)
#    print('da^2c^+ = %.3f MHz'%(dXi_a2s[index_max]*1e-6))
#    print('a^+2a^2 = %.3f MHz'%(Xi4[0,index_max,0]*1e-6))
#    print('a^+ac^+c = %.3f MHz'%(Xip[1,index_max,0]*1e-6))
#    print('f_snail = %.3f GHz'%(Xi2[1,index_max,0]*1e-9))
#    print('f_mem = %.3f GHz'%(Xi2[0,index_max,0]*1e-9))

# AS A FUNCTION OF COUPLING
if 1==0:

        
    Xi2 = np.zeros((4, len(CcaVec)))
    Xi3 = np.zeros((4, len(CcaVec)))
    Xi4 = np.zeros((4, len(CcaVec)))
    check_Xi2 = np.zeros((4, len(CcaVec)))
    Xi_pa2pb = np.zeros(len(CcaVec))
    Xi_ac = np.zeros(len(CcaVec))
    Xi_bc = np.zeros(len(CcaVec))
    resx = np.zeros(len(CcaVec))
    resy = np.zeros(len(CcaVec))
    comp0 = np.zeros((4, len(CcaVec)))
    comp1 = np.zeros((4, len(CcaVec)))
    comp2 = np.zeros((4, len(CcaVec)))
    comp3 = np.zeros((4, len(CcaVec)))
    for kk, xx in enumerate(ECcaVec):
        _res  = c.get_freqs_kerrs(particulars=[(1,1,2), (1,1,3,3), (2,2,3,3)], return_components=True, ECca=xx)
        res1, res2, Xi2s, Xi3s, Xi4s, Xi_p, P = _res
        Xi2[:, kk] = Xi2s
        Xi3[:, kk] = Xi3s
        Xi4[:, kk] = Xi4s
        Xi_pa2pb[kk] = Xi_p[0]
        Xi_ac[kk] = 4*Xi_p[1]
        Xi_bc[kk] = 4*Xi_p[2]
        resx[kk] = res1[0]
        resy[kk] = res1[1]
        comp0[:, kk] = (P.T)[0] 
        comp1[:, kk] = (P.T)[1] 
        comp2[:, kk] = (P.T)[2] 
        comp3[:, kk] = (P.T)[3] 
 