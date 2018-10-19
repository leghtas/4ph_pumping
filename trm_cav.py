# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:11:49 2016

@author: leghtas
"""

import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import circuit
import circuit_trm_cav as cp2s
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
w = 9.6e9*2*pi

wa, Za = [8e9*2*np.pi, 90]

Cc = 3.7*1e-15 #F

_, _, EJ = circuit.get_E_from_w(1, 1, LJ) #dLJ should be 0 to compute non-linearities !!!!!

#EJ, LJ, I0 = circuit.convert_EJ_LJ_I0(I0=I0)


c = cp2s.CircuitPump2Snail(w, EJ, wa, Za, Cc) 

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
    particulars = [(0,0,1,1)]
    factors_particulars = np.array([4])
        
    Xi2 = np.zeros((len(phiVec), max_solutions, n_modes))
    Xi3 = np.zeros((len(phiVec), max_solutions, n_modes))
    Xi4 = np.zeros((len(phiVec), max_solutions, n_modes))
    

    n_particulars = len(particulars)
    Xip = np.zeros((len(phiVec), max_solutions, n_particulars))

    comp = np.zeros((len(phiVec), max_solutions, n_modes, n_modes))
    for kk, xx in enumerate(phiVec):
        _res  = c.get_freqs_kerrs(particulars=particulars, return_components=True, max_solutions=max_solutions, phi_ext_0=xx)
        res1, res2, Xi2s, Xi3s, Xi4s, Xips, P= _res
        Xi2[kk] = Xi2s
        Xi3[kk] = Xi3s
        Xi4[kk] = 2*Xi4s
        Xip[kk] = Xips
        comp[kk] = np.moveaxis(P, 1, -1)
        
    Xi2 = np.moveaxis(Xi2, -1, 0)
    Xi3 = np.moveaxis(Xi3, -1, 0)
    Xi4 = np.moveaxis(Xi4, -1, 0)
    Xip = np.moveaxis(Xip*factors_particulars, -1, 0)

# PLOT
if 1==1:
    colors = ['b', 'r', 'y', 'g', 'o']
    fig0, ax0 = plt.subplots(figsize=(12,6))
    for ii, f in enumerate(Xi2):
        ax0.plot(phiVec/2/pi, f/1e9, '.', label= 'f'+str(ii), color = colors[ii])
#    ax0.plot(phiVec/2/pi, Xi2[2,:]/1e9, '.', label= 'f2')
#    ax0.plot(phiVec/2/pi, Xi2[3,:]/1e9, '.', label= 'f3')
    ax0.legend()
    ax0.set_ylabel('GHz')
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
    
    
# Plot particular
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


    print('\nf_cav = %.3f GHz'%(Xi2[1,0,0]/1e9))
    print('Kerr_cav = %.3f MHz'%(Xi4[1,0,0]/1e6))
    
    print('\ncross_Kerr = %.3f MHz'%(Xip[0,0]/1e6))
    
    print('\nf_trm = %.3f GHz'%(Xi2[0,0,0]/1e9))
    print('Kerr_trm = %.3f MHz\n'%(Xi4[0,0,0]/1e6))

# PLOT
if 1==0:
    fig0, ax0 = plt.subplots(figsize=(12,6))
    ax0.plot(phiVec/2/pi, Xi2[0,:]/1e9, '.', label= 'f0')    
    ax0.legend()
    ax0.set_ylabel('GHz')
    
    fig00, ax00 = plt.subplots(3,2,figsize=(12,6))
    ax00[0,0].plot(phiVec/2/pi, comp[0, 0,:], '.', label= 'f0/pa')    

    
    print('\nf0 = %.3f GHz\n'%(Xi2[0,:][0]/1e9))
    
    fig, ax = plt.subplots(4, 4, figsize=(16,8))
    ax[0,0].plot(phiVec/2/pi, Xi2[0,:]/1e9, '.', label= 'f0')
    ax[0,0].set_title('freq')
    ax[0,0].legend
    ax[0,1].plot(phiVec/2/pi, Xi3[0,:]/1e6)
    ax[0,1].set_title('c3')
    ax[0,2].plot(phiVec/2/pi, Xi4[0,:]/1e6)    
    ax[0,2].set_title('c4')

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
        
# PLOT
if 1==0:

    
    fig0, ax0 = plt.subplots(figsize=(12,6))
    ax0.plot(CcaVec, Xi2[0,:]/1e9, '.', label= 'f0')    
    ax0.plot(CcaVec, Xi2[1,:]/1e9, '.', label= 'f1')
    ax0.plot(CcaVec, Xi2[2,:]/1e9, '.', label= 'f2')
    ax0.plot(CcaVec, Xi2[3,:]/1e9, '.', label= 'f3')
    ax0.legend()
    ax0.set_ylabel('GHz')
    
    fig00, ax00 = plt.subplots(2,2,figsize=(12,6))
    ax00[0,0].plot(CcaVec, comp0[0,:], '.', label= 'f0/pa')    
    ax00[0,0].plot(CcaVec, comp0[1,:], '.', label= 'f0/pb')    
    ax00[0,0].plot(CcaVec, comp0[2,:], '.', label= 'f0/pc')    
    ax00[0,0].plot(CcaVec, comp0[3,:], '.', label= 'f0/pca')   
    
    ax00[1,0].plot(CcaVec, comp1[0,:], '.', label= 'f1/pa')    
    ax00[1,0].plot(CcaVec, comp1[1,:], '.', label= 'f1/pb')    
    ax00[1,0].plot(CcaVec, comp1[2,:], '.', label= 'f1/pc')    
    ax00[1,0].plot(CcaVec, comp1[3,:], '.', label= 'f1/pca')   
    
    ax00[0,1].plot(CcaVec, comp2[0,:], '.', label= 'f2/pa')    
    ax00[0,1].plot(CcaVec, comp2[1,:], '.', label= 'f2/pb')    
    ax00[0,1].plot(CcaVec, comp2[2,:], '.', label= 'f2/pc')    
    ax00[0,1].plot(CcaVec, comp2[3,:], '.', label= 'f2/pca')   
    
    ax00[1,1].plot(CcaVec, comp3[0,:], '.', label= 'f3/pa')    
    ax00[1,1].plot(CcaVec, comp3[1,:], '.', label= 'f3/pb')    
    ax00[1,1].plot(CcaVec, comp3[2,:], '.', label= 'f3/pc')    
    ax00[1,1].plot(CcaVec, comp3[3,:], '.', label= 'f3/pca')   

    ax00[0,0].legend()
    ax00[0,1].legend()
    ax00[1,0].legend()    
    ax00[1,1].legend()
    
    print('\nf0 = %.3f GHz\n'%(Xi2[0,:][0]/1e9)+'f1 = %.3f GHz\n'%(Xi2[1,:][0]/1e9)+'f2 = %.3f GHz\n'%(Xi2[2,:][0]/1e9)+'f3 = %.3f GHz\n'%(Xi2[3,:][0]/1e9))
    
    fig, ax = plt.subplots(4, 4, figsize=(16,8))
    ax[0,0].plot(CcaVec, Xi2[0,:]/1e9, '.', label= 'f0')
    ax[1,0].plot(CcaVec, Xi2[1,:]/1e9, '.', label= 'f1')
    ax[2,0].plot(CcaVec, Xi2[2,:]/1e9, '.', label= 'f2')
    ax[3,0].plot(CcaVec, Xi2[3,:]/1e9, '.', label= 'f3')
    ax[0,0].set_title('freq')

    ax[0,0].legend()
    ax[1,0].legend()
    ax[2,0].legend()
    ax[3,0].legend()
    
    ax[0,1].plot(CcaVec, Xi3[0,:]/1e6)    
    ax[1,1].plot(CcaVec, Xi3[1,:]/1e6)
    ax[2,1].plot(CcaVec, Xi3[2,:]/1e6)
    ax[3,1].plot(CcaVec, Xi3[3,:]/1e6)
    ax[0,1].set_title('c3')
    
    ax[0,2].plot(CcaVec, Xi4[0,:]/1e6)    
    ax[1,2].plot(CcaVec, Xi4[1,:]/1e6)
    ax[2,2].plot(CcaVec, Xi4[2,:]/1e6)
    ax[3,2].plot(CcaVec, Xi4[3,:]/1e6)
    ax[0,2].set_title('c4')


if 1==0:
    Nx = 21
    Uab0 = np.zeros((Nx, Nx))
    Uab = np.zeros((Nx, Nx))
    Uac0 = np.zeros((Nx, Nx))
    Uac = np.zeros((Nx, Nx))
    Ubc0 = np.zeros((Nx, Nx))
    Ubc = np.zeros((Nx, Nx))
    Tab0 = np.zeros((Nx, Nx))
    Tab = np.zeros((Nx, Nx))
    Tac0 = np.zeros((Nx, Nx))
    Tac = np.zeros((Nx, Nx))
    Tbc0 = np.zeros((Nx, Nx))
    Tbc = np.zeros((Nx, Nx))
    phi_ext_s_0 = np.pi/2
    phi_ext_l_0 = np.pi
    U = c.get_U(phi_ext_s_0=phi_ext_s_0, phi_ext_l_0=phi_ext_l_0)
    T = c.get_T(phi_ext_s_0=phi_ext_s_0, phi_ext_l_0=phi_ext_l_0)
    res1, res2, P, U2 = c.get_normal_mode_frame(phi_ext_s_0=phi_ext_s_0,
                                                phi_ext_l_0=phi_ext_l_0)
    x1, _y, _z = (nl.inv(P)).dot(np.array([np.pi, 0, 0]))
    _x, y1, _z = (nl.inv(P)).dot(np.array([0, np.pi, 0]))
    x_grid = np.linspace(-x1, x1, Nx)
    x_grid0 = np.linspace(-np.pi, np.pi, Nx)
    y_grid = np.linspace(-x1, x1, Nx)
    y_grid0 = np.linspace(-np.pi, np.pi, Nx)
    for ikk, kk in enumerate(x_grid):
        for iii, ii in enumerate(y_grid):
            Uab[ikk, iii] = U(np.array([kk, ii, 0]), P=P)
            Uac[ikk, iii] = U(np.array([kk, 0, ii]), P=P)
            Ubc[ikk, iii] = U(np.array([0, kk, ii]), P=P)
            Tab[ikk, iii] = T(np.array([kk, ii, 0]), P=P)
            Tac[ikk, iii] = T(np.array([kk, 0, ii]), P=P)
            Tbc[ikk, iii] = T(np.array([0, kk, ii]), P=P)
#            Tbc[ikk, iii] = T(np.array([0, 1e8*kk,1e8*ii]), P=nl.inv(P))
    for ikk, kk in enumerate(x_grid0):
        for iii, ii in enumerate(y_grid0):
            Uab0[ikk, iii] = U(np.array([kk, ii, 0]))
            Uac0[ikk, iii] = U(np.array([kk, 0, ii]))
            Ubc0[ikk, iii] = U(np.array([0, kk, ii]))
            Tab0[ikk, iii] = T(np.array([kk, ii, 0]))
            Tac0[ikk, iii] = T(np.array([kk, 0, ii]))
            Tbc0[ikk, iii] = T(np.array([0, kk, ii]))
#            Tbc0[ikk, iii] = T(np.array([kk, 0, ii]))
    fig2, ax2 = plt.subplots(2,3)
    ax2[0,0].pcolor(x_grid0, y_grid0, Uab0)
    ax2[1,0].pcolor(x_grid, y_grid, Uab)
    ax2[0,1].pcolor(x_grid0, y_grid0, Uac0)
    ax2[1,1].pcolor(x_grid, y_grid, Uac)
    ax2[0,2].pcolor(x_grid0, y_grid0, Ubc0)
    ax2[1,2].pcolor(x_grid, y_grid, Ubc)
    ax2[0,0].axis('equal')
    ax2[0,1].axis('equal')
    ax2[1,0].axis('equal')
    ax2[1,1].axis('equal')
    ax2[0,2].axis('equal')
    ax2[1,2].axis('equal')

    fig3, ax3 = plt.subplots(2,3)
    ax3[0,0].pcolor(x_grid0, y_grid0, Tab0)
    ax3[1,0].pcolor(x_grid, y_grid, Tab)
    ax3[0,1].pcolor(x_grid0, y_grid0, Tac0)
    ax3[1,1].pcolor(x_grid, y_grid, Tac)
    ax3[0,2].pcolor(x_grid0, y_grid0, Tbc0)
    ax3[1,2].pcolor(x_grid, y_grid, Tbc)
    ax3[0,0].axis('equal')
    ax3[0,1].axis('equal')
    ax3[1,0].axis('equal')
    ax3[1,1].axis('equal')
    ax3[0,2].axis('equal')
    ax3[1,2].axis('equal')
    #ax2[1].plot(phi_ext_sweep/np.pi, gradUval_)

if 1==0:
    fig, ax = plt.subplots(3, figsize=(12,8), sharex=True)
    a=0.755e-3/3
    ax[0].plot(a*phi_ext_sweep/np.pi, f[0,:]/1e9)
    ax[1].plot(a*phi_ext_sweep/np.pi, f[1,:]/1e9)
    ax[2].plot(a*phi_ext_sweep/np.pi, f[2,:]/1e9)

    ### data ###
    folder = r'../no_squid/analyzed_data/'

    filename_readout = r'sweep_DC_specVNA_DC2_in2outC_004.dat_'
    readout_freq = np.load(folder+filename_readout+'freq.npy')
    readout_flux = np.load(folder+filename_readout+'flux.npy')

    filename_buffer = r'sweep_DC_specVNA_DC2_in4outD_003.dat_'
    buffer_freq = np.load(folder+filename_buffer+'freq_fit.npy')
    buffer_flux = np.load(folder+filename_buffer+'flux.npy')

    filename_mem = r'VNA_sweep_DC_sweep_pump_freq_DC2_in2_outC_pump6_002.dat_'
    mem_freq = np.load(folder+filename_mem+'freq_fit.npy')
    mem_flux = np.load(folder+filename_mem+'flux.npy')

    offs = 1.2e-4
    ax[0].plot(mem_flux+1e-4-offs, mem_freq, 'o')
    ax[1].plot(buffer_flux+1e-4- offs, buffer_freq, 'o')
    ax[2].plot(readout_flux-offs, readout_freq/1e9, 'o')
    ax[2].set_xlabel('DC (V)')
    ax[0].set_ylabel('mem freq')
    ax[1].set_ylabel('buff freq')
    ax[2].set_ylabel('readout freq')


#fa = []
#fb = []
#
#for phi_ext_s in phi_ext_sweep:
#    H = c.getH_overhbar_two_mode(phi_ext_l=phi_ext_s)
#    E = H.eigenenergies()
#    fa.append(E[1]-E[0])
#    fb.append(E[2]-E[0])
#
#fig, ax = plt.subplots()
#ax.plot(phi_ext_sweep/np.pi, [f/2/np.pi/1e9 for f in fa])
#ax.plot(phi_ext_sweep/np.pi, [f/2/np.pi/1e9 for f in fb])

#c.pltElevels('phi_ext_s', phi_ext_sweep,
#             phiVec=np.linspace(-pi, pi, 101),
#             phi_ext_s_0=0, phi_ext_l_0=0, n_g_0=0)

#
# c.pltElevels('n_g', ng_sweep,
#             phiVec=np.linspace(-pi, pi, 101),
#             phi_ext_s_0=pi, phi_ext_l_0=0, n_g_0=0)

#fig, ax = plt.subplots()
#for ii in range(len(phi_ext_sweep)):
#    ax.plot((np.diff(Elevels[ii,:])[0:298]/1e9/2/np.pi))
#ax.plot((np.diff(Elevels[0,:])[0:70]/1e9/2/np.pi))

#for ii in range(29):
#    ax.plot(np.diff(Elevels)[:,ii]/1e9/2/np.pi)