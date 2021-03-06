# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:11:49 2016

@author: leghtas
"""

import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import circuit
import circuit_pumping2_snail as cp2s
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


wa, Za = [5.8e9*2*np.pi, 120]
wb, Zb = [9.6e9*2*np.pi, 120]#[9.6e9*2*np.pi, 120]

Cca, Ccb, Cc = [(252)*1e-15,(252)*1e-15, 25e-15]

LJ, dLJ, alpha, n = [2000e-12, 500e-12, 0.33, 3] #dLJ, linear part of inductance for sna



_, _, EJ = circuit.get_E_from_w(1, 1, LJ+dLJ) #dLJ should be 0 to compute non-linearities !!!!!

#EJ, LJ, I0 = circuit.convert_EJ_LJ_I0(I0=I0)


c = cp2s.CircuitPump2Snail(wa, Za, wb, Zb, Cca, Ccb, Cc, EJ, alpha, n) 

min_phi = 0*2*pi
max_phi = 1*2*pi
Npts = 101
phiVec = np.linspace(min_phi, max_phi, Npts)
ng_sweep = np.linspace(-1, 1, 21)

min_Cca = 1e-15
max_Cca = 100000e-15
Npts = 101
CcaVec = np.linspace(min_Cca, max_Cca, Npts)
ECcaVec = e**2/2/CcaVec


#Plot potential
if 1==0:
    fig, ax = plt.subplots(figsize = (12,12))
    ax.set_xlabel(r'$\varphi_r$', fontsize = fs)
    ax.set_ylabel(r'$\varphi_s$', fontsize = fs)
    min_i = 0
    max_i = 10
    def update(i):
        Phi_ext = np.linspace(0,2*np.pi, max_i-min_i)
        phi_ext= Phi_ext[i]
        U = c.get_U(phi_ext_0=phi_ext)
        shape=(40,20)
        Ps = np.linspace(-8*np.pi, 8*np.pi, shape[0])
        Pr = np.linspace(-8*np.pi, 8*np.pi, shape[1])

        U_color = np.empty(shape)
        for ii, ps in enumerate(Ps):
            for jj, pr in enumerate(Pr):
                U_color[ii, jj] = U(np.array([ps, pr]))

        ax.pcolor(Pr, Ps, U_color, vmin=vmin, vmax=vmax)
        return ax
#        move_figure(fig, -1900, 10)

    anim = FuncAnimation(fig, update, frames=np.arange(min_i, max_i), interval=200)
#    if len(sys.argv) > 1 and sys.argv[1] == 'save':
    anim.save('line.html', dpi=80, writer='imagemagick')

# Get freqs and Kerrs v.s. flux
if 1==1:
    
    Xi2 = np.zeros((4, len(phiVec)))
    Xi3 = np.zeros((4, len(phiVec)))
    Xi4 = np.zeros((4, len(phiVec)))
    check_Xi2 = np.zeros((4, len(phiVec)))
    Xi_pa2pb = np.zeros(len(phiVec))
    Xi_ac = np.zeros(len(phiVec))
    Xi_bc = np.zeros(len(phiVec))
    resx = np.zeros(len(phiVec))
    resy = np.zeros(len(phiVec))
    comp = np.zeros((4, 4, len(phiVec)))
    for kk, xx in enumerate(phiVec):
        _res  = c.get_freqs_kerrs(particulars=[(1,1,2), (1,1,3,3), (2,2,3,3)], return_components=True, phi_ext_0=xx)
        res1, res2, Xi2s, Xi3s, Xi4s, Xi_p, P= _res
        Xi2[:, kk] = Xi2s
        Xi3[:, kk] = Xi3s
        Xi4[:, kk] = Xi4s
        Xi_pa2pb[kk] = Xi_p[0]
        Xi_ac[kk] = 4*Xi_p[1]
        Xi_bc[kk] = 4*Xi_p[2]
        resx[kk] = res1[0]
        resy[kk] = res1[1]
        comp[:, :, kk] = (P.T) 


# PLOT
if 1==1:
    fig0, ax0 = plt.subplots(figsize=(12,6))
    ax0.plot(phiVec/2/pi, Xi2[0,:]/1e9, '.', label= 'f0')    
    ax0.plot(phiVec/2/pi, Xi2[1,:]/1e9, '.', label= 'f1')
    ax0.plot(phiVec/2/pi, Xi2[2,:]/1e9, '.', label= 'f2')
    ax0.plot(phiVec/2/pi, Xi2[3,:]/1e9, '.', label= 'f3')
    ax0.legend()
    ax0.set_ylabel('GHz')
    
    fig00, ax00 = plt.subplots(2,2,figsize=(12,6))
    ax00[0,0].plot(phiVec/2/pi, comp[0, 0,:], '.', label= 'f0/pa')    
    ax00[0,0].plot(phiVec/2/pi, comp[0, 1,:], '.', label= 'f0/pb')    
    ax00[0,0].plot(phiVec/2/pi, comp[0, 2,:], '.', label= 'f0/pc')    
    ax00[0,0].plot(phiVec/2/pi, comp[0, 3,:], '.', label= 'f0/pca')   
    ax00[0,0].plot(phiVec/2/pi, ((comp**2).sum(0)**0.5)[0], '.', label= 'norm') 

    ax00[1,0].plot(phiVec/2/pi, comp[1, 0,:], '.', label= 'f1/pa')    
    ax00[1,0].plot(phiVec/2/pi, comp[1, 1,:], '.', label= 'f1/pb')    
    ax00[1,0].plot(phiVec/2/pi, comp[1, 2,:], '.', label= 'f1/pc')    
    ax00[1,0].plot(phiVec/2/pi, comp[1, 3,:], '.', label= 'f1/pca')   
    ax00[1,0].plot(phiVec/2/pi, ((comp**2).sum(0)**0.5)[1], '.', label= 'norm') 

    ax00[0,1].plot(phiVec/2/pi, comp[2, 0,:], '.', label= 'f2/pa')    
    ax00[0,1].plot(phiVec/2/pi, comp[2, 1,:], '.', label= 'f2/pb')    
    ax00[0,1].plot(phiVec/2/pi, comp[2, 2,:], '.', label= 'f2/pc')    
    ax00[0,1].plot(phiVec/2/pi, comp[2, 3,:], '.', label= 'f2/pca')   
    ax00[0,1].plot(phiVec/2/pi, ((comp**2).sum(0)**0.5)[2], '.', label= 'norm') 

    ax00[1,1].plot(phiVec/2/pi, comp[3, 0,:], '.', label= 'f3/pa')    
    ax00[1,1].plot(phiVec/2/pi, comp[3, 1,:], '.', label= 'f3/pb')    
    ax00[1,1].plot(phiVec/2/pi, comp[3, 2,:], '.', label= 'f3/pc')    
    ax00[1,1].plot(phiVec/2/pi, comp[3, 3,:], '.', label= 'f3/pca')   
    ax00[1,1].plot(phiVec/2/pi, ((comp**2).sum(0)**0.5)[3], '.', label= 'norm') 

    ax00[0,0].legend()
    ax00[0,1].legend()
    ax00[1,0].legend()    
    ax00[1,1].legend()
    
    print('\nf0 = %.3f GHz\n'%(Xi2[0,:][0]/1e9)+'f1 = %.3f GHz\n'%(Xi2[1,:][0]/1e9)+'f2 = %.3f GHz\n'%(Xi2[2,:][0]/1e9)+'f3 = %.3f GHz\n'%(Xi2[3,:][0]/1e9))
    
    fig, ax = plt.subplots(4, 4, figsize=(16,8))
    ax[0,0].plot(phiVec/2/pi, Xi2[0,:]/1e9, '.', label= 'f0')
    ax[1,0].plot(phiVec/2/pi, Xi2[1,:]/1e9, '.', label= 'f1')
    ax[2,0].plot(phiVec/2/pi, Xi2[2,:]/1e9, '.', label= 'f2')
    ax[3,0].plot(phiVec/2/pi, Xi2[3,:]/1e9, '.', label= 'f3')
    ax[0,0].set_title('freq')

    ax[0,0].legend()
    ax[1,0].legend()
    ax[2,0].legend()
    ax[3,0].legend()
    
    ax[0,1].plot(phiVec/2/pi, Xi3[0,:]/1e6)    
    ax[1,1].plot(phiVec/2/pi, Xi3[1,:]/1e6)
    ax[2,1].plot(phiVec/2/pi, Xi3[2,:]/1e6)
    ax[3,1].plot(phiVec/2/pi, Xi3[3,:]/1e6)
    ax[0,1].set_title('c3')
    
    ax[0,2].plot(phiVec/2/pi, Xi4[0,:]/1e6)    
    ax[1,2].plot(phiVec/2/pi, Xi4[1,:]/1e6)
    ax[2,2].plot(phiVec/2/pi, Xi4[2,:]/1e6)
    ax[3,2].plot(phiVec/2/pi, Xi4[3,:]/1e6)
    ax[0,2].set_title('c4')

    fig2, ax2 = plt.subplots(2,2, figsize=(16,8))
    ax2[0,0].plot(phiVec/2/pi, Xi_pa2pb/1e6, '.', label= '$a^2b^{+}$')
    
    dXi_pa2pb=np.diff(Xi_pa2pb)/(phiVec[1]-phiVec[0])*pi/10
    dphiVec = (phiVec[1:]+phiVec[:-1])/2
    
    ax2[0,0].plot(dphiVec/2/pi, dXi_pa2pb/1e6, '.', label= r'$\frac{\pi}{10}*da^2b^{+}$')
    ax2[0,0].legend()
    ax2[0,0].set_ylabel('MHz')
    
    Xi4_1 = (Xi4[1,1:]+Xi4[1,:-1])/2
    ax2[1,0].plot(dphiVec/2/pi, Xi4_1/1e6, '.', label= '$a^2a^{+2}$')
    ax2[1,0].plot([min(dphiVec/2/pi), max(dphiVec/2/pi)], [0,0])
    ax2[1,0].legend()
    ax2[1,0].set_ylabel('MHz')

    ax2[1,1].plot(phiVec/2/pi, Xi_ac/1e6, '.', label= r'$a^{+}ac^{+}c$')
    ax2[1,1].plot(phiVec/2/pi, Xi_bc/1e6, '.', label= r'$b^{+}bc^{+}c$')
    ax2[1,1].legend()
    
    ax2[0,1].plot(dphiVec/2/pi, dXi_pa2pb/Xi4_1, '.', label= '$a^2b^{+}/a^2a^{+2}$')
#    ax[3,1].plot(phiVec/2/pi, Xi4[1,:]/1e6, label= '$b^2b^{+2}$')
    ax2[0,1].legend()

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