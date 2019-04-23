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

#Z, w, LJ, N, alpha, n = [50, 8*1e9*2*np.pi, 60e-12, 1, 0.29, 3]
#w0, wa, LJ, N, alpha, n = [10.53*1e9*2*np.pi, 4.45*1e9*2*np.pi, 0.1e-9, 20, 0.1, 3]
w0, wa, LJ, N, alpha, n = [10.53*1e9*2*np.pi, 4.45*1e9*2*np.pi, 0.1e-9, 20, 0.1, 3]
#w0, wa, LJ, N, alpha, n = [14.89*1e9*2*np.pi, 5.64*1e9*2*np.pi, 0.1e-9, 20, 0.1, 3]
#LJ, N, alpha, n = [40e-12, 20, 0.01, 3]

LJeq = N*(1/(1/(LJ*n)+ 1/(LJ/alpha)))
EC, EL, EJeq = circuit.get_E_from_w0_wa_LJ(w0, wa, LJeq)

#EC, EL, _ = circuit.get_E_from_w(4.45*1e9*2*np.pi, 50, 1)
#EC2, EL2, _ = circuit.get_E_from_w(6*1e9*2*np.pi,50, 1)

#EJ, LJ, I0 = circuit.convert_EJ_LJ_I0(I0=I0)

EJ = phi0**2/LJ
ECJ = EC*3
c = cspa.CircuitSPA(EL, EJ, EC, ECJ, N, n, alpha)

min_phi = -2*pi
max_phi = 2*pi
Npts = 201
phiVec = np.linspace(min_phi, max_phi, Npts)
ng_sweep = np.linspace(-1, 1, 21)

Xi2 = np.zeros((2, len(phiVec)))
Xi2_ = np.zeros((2, len(phiVec)))
Xi3 = np.zeros((2, len(phiVec)))
Xi4 = np.zeros((2, len(phiVec)))
check_Xi2 = np.zeros((2, len(phiVec)))
resx = np.zeros(len(phiVec))
resy = np.zeros(len(phiVec))


if 1==0:
    U = c.get_U(phi_ext_0=np.pi)
    Uarray = np.zeros((Npts, Npts))
    for i1, p1 in enumerate(phiVec):
        for i2, p2 in enumerate(phiVec):
            Uarray[i1, i2] = U([p1, p2])/2/np.pi/1e9
    fig, ax = plt.subplots()
    ax.pcolor(phiVec, phiVec, Uarray)
    fig, ax = plt.subplots()
    ax.plot(phiVec, Uarray[100,:])
    ax.plot(phiVec, Uarray[:,100])
# Get freqs and Kerrs v.s. flux
if 1==1:
    for kk, xx in enumerate(phiVec):
        fs = c.get_freqs_only(pext=xx)
        _res = c.get_freqs_kerrs(pext=xx)
        res1s, res2, Xi2s, Xi3s, Xi4s, Xips, P = _res
        Xi2[:, kk] = fs
        Xi2_[:, kk] = Xi2s
#        break

# PLOT
if 1==1:
    fig, ax = plt.subplots(2, 2, figsize=(16,8))
#    ax[0].plot(phi_ext_sweep/np.pi, resx/np.pi)
#    ax[0].plot(phi_ext_sweep/np.pi, resy/np.pi)
#    ax[0].plot(phi_ext_sweep/np.pi, resz/np.pi)
    ax[0,0].plot(phiVec/2/pi, Xi2[0,:]/1e9)
    ax[0,1].plot(phiVec/2/pi, Xi2_[0,:]/1e9)
    ax[1,0].plot(phiVec/2/pi, Xi2[1,:]/1e9)
    ax[1,1].plot(phiVec/2/pi, Xi2_[1,:]/1e9)
    