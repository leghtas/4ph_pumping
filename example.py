# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:11:49 2016

@author: leghtas
"""

import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import circuit
import circuit_SnailPA_capa as cspa
import scipy.linalg as sl
import numpy.linalg as nl
import numdifftools as nd
from scipy.misc import factorial
import matplotlib
from matplotlib.animation import FuncAnimation
import sys

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

Z, w, LJ, NN, alpha, n = [50, 6.8*1e9*2*np.pi, 31e-12, 1, 0.23, 3]
#Z, w, LJ, NN, alpha, n = [50, 8*1e9*2*np.pi, 47e-12, 20, 0.09, 3]
#LJ, NN, alpha, n = [40e-12, 20, 0.01, 3]



EC, EL, EJ = circuit.get_E_from_w(w, Z, LJ)

#EJ, LJ, I0 = circuit.convert_EJ_LJ_I0(I0=I0)


c = cspa.CircuitSnailPA(EC, EL, EJ, alpha, n, NN=NN, printParams=True)

min_phi = -4*pi
max_phi = 4*pi
Npts = 201
phiVec = np.linspace(min_phi, max_phi, Npts)
ng_sweep = np.linspace(-1, 1, 21)

Xi2 = np.zeros((2, len(phiVec)))
Xi3 = np.zeros((2, len(phiVec)))
Xi4 = np.zeros((2, len(phiVec)))
check_Xi2 = np.zeros((2, len(phiVec)))
resx = np.zeros(len(phiVec))
resy = np.zeros(len(phiVec))

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
    for kk, xx in enumerate(phiVec):
        _res  = c.get_freqs_kerrs(phi_ext_0=xx)
        res1, res2, Xi2s, Xi3s, Xi4s, check_Xi2s = _res
        Xi2[:, kk] = Xi2s
        Xi3[:, kk] = Xi3s
        Xi4[:, kk] = Xi4s
        check_Xi2[:, kk] = check_Xi2s
        resx[kk] = res1[0]
        resy[kk] = res1[1]


# PLOT
if 1==1:
    fig, ax = plt.subplots(2, 4, figsize=(16,8))
#    ax[0].plot(phi_ext_sweep/np.pi, resx/np.pi)
#    ax[0].plot(phi_ext_sweep/np.pi, resy/np.pi)
#    ax[0].plot(phi_ext_sweep/np.pi, resz/np.pi)
    ax[0,0].plot(phiVec/2/pi, Xi2[0,:]/1e9)
    ax[1,0].plot(phiVec/2/pi, Xi2[1,:]/1e9)
    ax[0,0].plot(phiVec/2/pi, check_Xi2[0,:]/1e9)
    ax[1,0].plot(phiVec/2/pi, check_Xi2[1,:]/1e9)

    ax[0,1].plot(phiVec/2/pi, np.abs(Xi3[0,:]/1e6))
    ax[1,1].plot(phiVec/2/pi, np.abs(Xi3[1,:]/1e6))
    ax[0,1].plot(np.array([min_phi, max_phi])/2/np.pi, [0,0])
    ax[1,1].plot(np.array([min_phi, max_phi])/2/np.pi, [0,0])

    ax[0,2].plot(phiVec/2/pi, Xi4[0,:]/1e6)
    ax[1,2].plot(phiVec/2/pi, Xi4[1,:]/1e6)
    ax[0,2].plot(np.array([min_phi, max_phi])/2/np.pi, [0,0])
    ax[1,2].plot(np.array([min_phi, max_phi])/2/np.pi, [0,0])

    fmin = min(Xi2[1,:]/1e9)
    fmax = max(Xi2[1,:]/1e9)
#    ax[0,0].set_ylim([4, 8])
    ax[1,3].semilogy(Xi2[1,:]/1e9, np.abs(Xi3[1,:]/Xi4[1,:]))
    ax[1,3].semilogy([fmin, fmax], [10,10])
    ax[1,3].semilogy([fmin, fmax], [100,100])
    #ax[1,3].set_ylim([5,np.max(np.abs(Xi3[1,:]/Xi4[1,:]))])

    ax[0,0].set_title('frequency (GHz)')
    ax[0,1].set_title('$a^2a^{\dag}$ (MHz)')
    ax[0,2].set_title('$a^2a^{\dag 2}$ (MHz)')
    ax[0,3].set_title('$a^2a^{\dag}/a^2a^{\dag 2}$')
    ax[0,0].set_ylabel('Resonator')
    ax[1,0].set_ylabel('Plasma')

#    fig2, ax2 = plt.subplots(2)
#    ax2[0].scatter(k[1,:]/1e9, k[0,:]/1e9, 10, label='0 photon')
#    ax2[0].scatter(k[1,:]/1e9+Xi4[3, :]/1e9, k[0,:]/1e9+Xi22[:]/1e9, 10, label='1 photon')
#    ax2[0].scatter(wb/2/pi/1e9, wa/2/pi/1e9, marker='x')
#    ax2[0].set_ylabel('MEMORY freq (GHz)')
#    ax2[0].set_xlabel('BUFFER freq (GHz)')
#    ax2[1].plot(phi_ext_sweep*BL/2/pi, k[1,:]/1e9, label='0 photon')
#    ax2[1].plot(phi_ext_sweep*BL/2/pi, k[1,:]/1e9+Xi4[3, :]/1e9, label='1 photon')
#    ax2[1].legend()
#    ax2[0].legend()

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