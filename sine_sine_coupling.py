# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:11:49 2016

@author: leghtas
"""

import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import circuit
import scipy.linalg as sl
import numpy.linalg as nl
import numdifftools as nd
from scipy.misc import factorial

plt.close('all')
Phi0 = sc.value('mag. flux quantum')
e = sc.elementary_charge
phi0 = Phi0/2/np.pi  # Phi_0=h/(2*e)
pi = sc.pi
hbar = sc.hbar
h = sc.h

wa = 2*pi*4e9  # res frequency (Hz/rad) HFSS 6.13
wb = 2*pi*6.2e9  # res frequency (Hz/rad) HFSS 3.31
wc = 2*pi*7.7445e9  # res frequency (Hz/rad) HFSS 7.3
Za = 130.  # res impedance in Ohm (cavity)
Zb = 130.  # res impedance in Ohm (qubit)
Zc = 50.
LJ1 = 1.0 * 15e-9  # Josephson inductance, each junction has 2*LJ
LJ2 = 1.1 * 15e-9 * 1.0
# ECa = h*200*1e6
# EJa = 40*ECa
# ELa = EJa/1e4
# wa, Za, LJ = circuit.get_w_Z_LJ_from_E(EC, EJ, EL)
ECa, ELa, EJ1 = circuit.get_E_from_w(wa, Za, LJ1)
ECb, ELb, EJ2 = circuit.get_E_from_w(wb, Zb, LJ2)
ECc, ELc, EJ = circuit.get_E_from_w(wc, Zc, LJ1)
na, nb, nc = 6, 6, 6
eps = 0 * ECc/100
c = circuit.Circuit(ECa, ELa, ECb, ELb, EJ1, EJ2, na, nb,
                    ECc=ECc-eps, ELc=ELc, Ecoup=12*ECa+eps, nc=nc,
                    printParams=True)

phi_ext_sweep = np.linspace(pi-2*pi/10, pi+2*pi/10, 101)
phiVec = np.linspace(-4*0.5*2*pi, 4*0.5*2*pi, 101)
ng_sweep = np.linspace(-1, 1, 21)

f = np.zeros((3, len(phi_ext_sweep)))
k = np.zeros((3, len(phi_ext_sweep)))
Xi3 = np.zeros((5, len(phi_ext_sweep)))
Xi4 = np.zeros((5, len(phi_ext_sweep)))
resx = np.zeros(len(phi_ext_sweep))
resy = np.zeros(len(phi_ext_sweep))
resz = np.zeros(len(phi_ext_sweep))
gradUval = np.zeros(len(phi_ext_sweep))
gradUval_ = np.zeros(len(phi_ext_sweep))

if 1==1:
    for kk, phi_ext in enumerate(phi_ext_sweep):
        res1, res2, fs, fs_diff, Xi3s, Xi4s, coeff = c.get_freqs_kerrs(phi_ext_s_0=1*phi_ext,
                                                              phi_ext_l_0=12*phi_ext+pi/4)
        f[:, kk] = np.sort(fs)
        k[:, kk]= np.sort(fs_diff[2:])
        Xi3[:, kk] = Xi3s
        Xi4[:, kk] = Xi4s
        resx[kk] = res1[0]
        resy[kk] = res1[1]
        resz[kk] = res1[2]

if 1==0:
    U = c.get_U(phi_ext_s_0=0, phi_ext_l_0=np.pi)
    res = c.get_normal_mode_frame(phi_ext_s_0=0, phi_ext_l_0=np.pi)
    res1, res2, P, w2 = res
    [x0, y0, z0] = (nl.inv(P)).dot([res1[0], res1[1], res1[2]])
    Ux = lambda x : U([x+x0, y0, z0], P=P)
    Uy = lambda y : U([x0, y+y0, z0], P=P)
    Uz = lambda z : U([x0, y0, z+z0], P=P)
    #
    UxVec = np.zeros(len(phi_ext_sweep))
    UyVec = np.zeros(len(phi_ext_sweep))
    UzVec = np.zeros(len(phi_ext_sweep))
    for ii, pp in enumerate(phi_ext_sweep):
        UxVec[ii], UyVec[ii], UzVec[ii]  = [Ux(pp), Uy(pp), Uz(pp)]

    fig, ax = plt.subplots()
    ax.plot(phi_ext_sweep, UxVec)
    ax.plot(phi_ext_sweep, UyVec)
    ax.plot(phi_ext_sweep, UzVec)
    taylor = Ux(0) + \
             nd.Derivative(Ux, n=2)(0)*(1/factorial(2))*(phi_ext_sweep)**2
    ax.plot(phi_ext_sweep, taylor,  'o')
    taylor = Uy(0) + \
             nd.Derivative(Uy, n=2)(0)*(1/factorial(2))*(phi_ext_sweep)**2
    ax.plot(phi_ext_sweep, taylor,  'o')
    taylor = Uz(0) + \
             nd.Derivative(Uz, n=2)(0)*(1/factorial(2))*(phi_ext_sweep)**2
    ax.plot(phi_ext_sweep, taylor,  'o')

if 1==1:
    fig, ax = plt.subplots(3, 3, figsize=(16,8))
#    ax[0].plot(phi_ext_sweep/np.pi, resx/np.pi)
#    ax[0].plot(phi_ext_sweep/np.pi, resy/np.pi)
#    ax[0].plot(phi_ext_sweep/np.pi, resz/np.pi)
    ax[0,0].plot(phi_ext_sweep/2/np.pi, f[0,:]/1e9)
    ax[1,0].plot(phi_ext_sweep/2/np.pi, f[1,:]/1e9)
    ax[2,0].plot(phi_ext_sweep/2/np.pi, f[2,:]/1e9)
    ax[0,0].plot(phi_ext_sweep/2/np.pi, k[0,:]/1e9)
    ax[1,0].plot(phi_ext_sweep/2/np.pi, k[1,:]/1e9)
    ax[2,0].plot(phi_ext_sweep/2/np.pi, k[2,:]/1e9)
    ax[0,1].plot(phi_ext_sweep/2/np.pi, Xi3[0,:]/1e6)
    ax[0,1].plot(phi_ext_sweep/2/np.pi, Xi3[2,:]/1e6)
    ax[1,1].plot(phi_ext_sweep/2/np.pi, Xi3[1,:]/1e6)
    ax[1,1].plot(phi_ext_sweep/2/np.pi, Xi3[3,:]/1e6)
    ax[0,2].plot(phi_ext_sweep/2/np.pi, Xi4[0,:]/1e6)
    ax[0,2].plot(phi_ext_sweep/2/np.pi, Xi4[2,:]/1e6)
    ax[1,2].plot(phi_ext_sweep/2/np.pi, Xi4[1,:]/1e6)
    ax[1,2].plot(phi_ext_sweep/2/np.pi, Xi4[3,:]/1e6)

    ax[0,0].set_title('frequency (GHz)')
    ax[0,1].set_title('Xi3 (MHz)')
    ax[0,2].set_title('Kerr (MHz)')
    ax[0,0].set_ylabel('MEMORY')
    ax[1,0].set_ylabel('BUFFER')
    ax[2,0].set_ylabel('READOUT')
    ax[2,0].set_xlabel('flux (/2pi)')
    ax[2,1].set_xlabel('flux (/2pi)')
    ax[2,2].set_xlabel('flux (/2pi)')

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


fig, ax = plt.subplots()
ax.plot(phi_ext_sweep/2/np.pi, f[0,:]/1e9, color='blue')
ax.set_xlabel('flux (/2pi)')
ax.set_ylabel('frequency (GHz)')
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