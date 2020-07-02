# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:11:49 2016

@author: leghtas
"""

import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import circuit
import circuitSPA as cspa
import scipy.linalg as sl
import numpy.linalg as nl
import numdifftools as nd
from scipy.misc import factorial
pi = np.pi
Phi0 = sc.value('mag. flux quantum')
e = sc.elementary_charge
phi0 = Phi0/2/np.pi  # Phi_0=h/(2*e)

plt.close('all')
#LJ = 1.32e-08
#I0 = 7.1*1e-6 #A EJ=phi0*I0

new_device = True
if new_device:
    # tune up procedure:
    #  wa: fix for desired working point: 7.5 GHz < 8
    # LJ = largest possible for acceptable Kerr
    # w0 = frequency of mode when LJ=0. Closest possible to wa, with acceptable LGeometric (i.e < nH)
    # alpha = tune asymetry of LJ*n and LJ/alpha for desired tunability
    w0, wa, LJ, N, alpha, n = [9*1e9*2*np.pi, 7.5*1e9*2*np.pi, 0.05e-9, 1, 0.2, 3]
# Low
# w0, wa, LJ, N, alpha, n = [15*1e9*2*np.pi, 7.5*1e9*2*np.pi, 0.043e-9, 1, 0.8, 1]
#w0, wa, LJ, N, alpha, n = [15*1e9*2*np.pi, 7.5*1e9*2*np.pi, 0.043e-9, 1, 0.1, 3]
measured_device = False
if measured_device:
    LJ1 = 0.35e-9*1
    LJ2 = 0.5e-9*1
    alpha = LJ1/LJ2
    w0, wa, LJ, N, alpha, n = [8.1*1e9*2*np.pi, 7.5*1e9*2*np.pi, LJ1, 1, alpha, 1]
# High
#w0, wa, LJ, N, alpha, n = [14.89*1e9*2*np.pi, 6.2*1e9*2*np.pi, 0.1e-9, 20, 0.1, 3]

LJeq = N*(1/(1/(LJ*n)+ 1/(LJ/alpha)))
print('LJeq = %s nH' % str(LJeq/1e-9))

EC, EL, EJeq = circuit.get_E_from_w0_wa_LJ(w0, wa, LJeq)
LG = phi0**2/EL
C = e**2/2/EC
print('LG = %s nH' % str(LG/1e-9))
print('Z = sqrt((LG+LJ)/C) = %s Ohm' % str(np.sqrt((LG+LJeq)/C)))
#EC, EL, _ = circuit.get_E_from_w(4.45*1e9*2*np.pi, 50, 1)
#EC2, EL2, _ = circuit.get_E_from_w(6*1e9*2*np.pi,50, 1)

#EJ, LJ, I0 = circuit.convert_EJ_LJ_I0(I0=I0)

EJ = phi0**2/LJ
ECJ = EC*10
c = cspa.CircuitSPA(EL, EJ, EC, ECJ, N, n, alpha)

min_phi = -2*pi+0*pi/10
max_phi = 2*pi-0*pi/10
Npts = 101
phiVec = np.linspace(min_phi, max_phi, Npts)
ng_sweep = np.linspace(-1, 1, 21)

Xi2 = np.zeros((2, len(phiVec)))
Xi2_ = np.zeros((2, len(phiVec)))
Xi3 = np.zeros((2, len(phiVec)))
Xi4 = np.zeros((2, len(phiVec)))
check_Xi2 = np.zeros((2, len(phiVec)))
resx = np.zeros(len(phiVec))
resy = np.zeros(len(phiVec))

# Get freqs and Kerrs v.s. flux
if 1==1:
    for kk, xx in enumerate(phiVec):
        _res = c.get_freqs_kerrs(pext=xx)
        res1s, res2, Xi2s, Xi3s, Xi4s, Xips, P = _res
        Xi2[:, kk] = Xi2s
        Xi3[:, kk] = Xi3s
        Xi4[:, kk] = Xi4s
#        break

# PLOT
if 1==1:
    fig, ax = plt.subplots(2, 4, figsize=(16,8))
#    ax[0].plot(phi_ext_sweep/np.pi, resx/np.pi)
#    ax[0].plot(phi_ext_sweep/np.pi, resy/np.pi)
#    ax[0].plot(phi_ext_sweep/np.pi, resz/np.pi)
    ax[0,0].plot(phiVec/2/pi, Xi2[0,:]/1e9)
    ax[1,0].plot(phiVec/2/pi, Xi2[1,:]/1e9)

    ax[0,1].plot(phiVec/2/pi, np.abs(Xi3[0,:]/1e6))
    ax[1,1].plot(phiVec/2/pi, np.abs(Xi3[1,:]/1e6))
    ax[0,1].plot(np.array([min_phi, max_phi])/2/np.pi, [0,0])
    ax[1,1].plot(np.array([min_phi, max_phi])/2/np.pi, [0,0])

    ax[0,2].plot(phiVec/2/pi, Xi4[0,:]/1e6)
    ax[1,2].plot(phiVec/2/pi, Xi4[1,:]/1e6)
    ax[0,2].plot(np.array([min_phi, max_phi])/2/np.pi, [0,0])
    ax[1,2].plot(np.array([min_phi, max_phi])/2/np.pi, [0,0])

    fmin = min(Xi2[0,:]/1e9)
    fmax = max(Xi2[0,:]/1e9)
    ax[0,3].semilogy(Xi2[0,:]/1e9, np.abs(Xi3[0,:]/Xi4[0,:])/2)
    ax[0,3].semilogy([fmin, fmax], [10,10])
    ax[0,3].semilogy([fmin, fmax], [100,100])
    try:
        ax[0,3].set_ylim([5,np.max(np.abs(Xi3[0,:]/Xi4[0,:])/2)])
    except:
        pass


    ax[1,0].set_title('frequency (GHz)')
    ax[1,1].set_title('$a^2a^{\dag}$ (MHz)')
    ax[1,2].set_title('$a^2a^{\dag 2}$ (MHz)')
    ax[1,3].set_title('$a^2a^{\dag}/a^2a^{\dag 2}$')
    ax[0,0].set_ylabel('SPA frequency (GHz)')
    ax[1,0].set_ylabel('Plasma frequency (GHz)')
    ax[1,0].set_xlabel('$\Phi_{\mathrm{ext}}/\Phi_0$')
    ax[1,3].set_xlabel('frequency (GHz)')
    ax[1,3].set_ylabel('$c_3/K$')
