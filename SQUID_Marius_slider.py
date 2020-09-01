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
from matplotlib.widgets import Slider, Button, RadioButtons


pi = np.pi
Phi0 = sc.value('mag. flux quantum')
e = sc.elementary_charge
phi0 = Phi0/2/np.pi  # Phi_0=h/(2*e)

plt.close('all')

wa, LJ, N, alpha, n = [8*1e9*2*np.pi, 0.1e-9, 1, 0.2, 3]

wa, LJ, N, alpha, n = [8.3*1e9*2*np.pi, 0.13e-9, 1, 0.25, 3]
# LJ of large junction
#wa, LJ, N, alpha, n = [8*1e9*2*np.pi, 0.3e-9, 1, 0.6, 1]

LJeq = N*(1/(1/(LJ*n)+ 1/(LJ/alpha)))
print('LJeq = %s nH' % str(LJeq/1e-9))

min_phi = -pi+0*pi/10
max_phi = pi-0*pi/10
Npts = 101
phiVec = np.linspace(min_phi, max_phi, Npts)
dPhi = phiVec[1]-phiVec[0]
noise = 100*1e-6

def houches(wa, LJ, N, alpha, n, phiVec):
    Z = 50
    LJeq = N*(1/(1/(LJ*n)+ 1/(LJ/alpha)))
    LG = (np.sqrt(LJeq**2+4*(Z/wa)**2) - LJeq**2)/2
    C = LG/Z**2
    EC = e**2/2/C

    ECJ = EC*10
    EJ = phi0**2/LJ
    EL = phi0**2/LG
    c = cspa.CircuitSPA(EL, EJ, EC, ECJ, N, n, alpha)

    Xi2 = np.zeros((2, len(phiVec)))
    Xi3 = np.zeros((2, len(phiVec)))
    Xi4 = np.zeros((2, len(phiVec)))

    for kk, xx in enumerate(phiVec):
        _res = c.get_freqs_kerrs(pext=xx)
        res1s, res2, Xi2s, Xi3s, Xi4s, Xips, P = _res
        Xi2[:, kk] = Xi2s
        Xi3[:, kk] = Xi3s
        Xi4[:, kk] = Xi4s

    kappaphi = np.diff(Xi2[0,:])/dPhi*noise

    return Xi2, Xi3, Xi4, kappaphi

def update(val):
    wa = s_wa.val*2*pi*1e9
    LJ = s_LJ.val*1e-9
    N = int(s_N.val)
    alpha = s_alpha.val
    n = int(s_n.val)

    Xi2, Xi3, Xi4, kappaphi = houches(wa, LJ, N, alpha, n, phiVec)
    axXi2.set_ydata(Xi2[0,:]/1e9)
    axXi3.set_ydata(Xi3[0,:]/1e6)
    axXi4.set_ydata(Xi4[0,:]/1e6)
    axkappaphi.set_ydata(kappaphi/1e3)

    fig.canvas.draw_idle()

def reset(event):
    s_wa.reset()
    s_LJ.reset()
    s_N.reset()
    s_alpha.reset()
    s_n.reset()


if 1:
    fig, ax = plt.subplots(1,4,figsize=(25,8))
    plt.subplots_adjust(left=0.05, bottom=0.35)

    Xi2, Xi3, Xi4, kappaphi = houches(wa, LJ, N, alpha, n, phiVec)

    axXi2, = ax[0].plot(phiVec/2/pi, Xi2[0,:]/1e9)
    axXi3, = ax[1].plot(phiVec/2/pi, Xi3[0,:]/1e6)
    axXi4, = ax[2].plot(phiVec/2/pi, Xi4[0,:]/1e6)
    axkappaphi, = ax[3].plot(phiVec[:-1]/2/pi, kappaphi/1e3)

    axcolor = 'white'
    ax_wa = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_LJ = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    ax_N = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_alpha = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_n = plt.axes([0.25, 0.2, 0.25, 0.03], facecolor=axcolor)
    s_wa = Slider(ax_wa, r'$\omega_a/2pi (GHz)$', 6, 12, valinit=wa/1e9/2/pi)
    s_LJ = Slider(ax_LJ, r'$L_J$ (nH)', 0.01, 1, valinit=LJ/1e-9)
    s_N = Slider(ax_N, r'$N$', 1, 10, valinit=N, valstep=1, valfmt='%1.0f')
    s_alpha = Slider(ax_alpha, r'$\alpha$', 0.1, 1, valinit=alpha)
    s_n = Slider(ax_n, r'$n$', 1, 5, valinit=n, valstep=1, valfmt='%1.0f')

    s_wa.on_changed(update)
    s_LJ.on_changed(update)
    s_N.on_changed(update)
    s_alpha.on_changed(update)
    s_n.on_changed(update)

#    resetax = plt.axes([0.05, 0.15, 0.1, 0.04])
#    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
#
#    button.on_clicked(reset)

    ax[0].grid(c='lightgrey')
#    ax[0].set_ylim([0,15])
    ax[0].set_ylabel(r'$\omega/2\pi$ (GHz)')
    ax[0].set_xlabel(r'$\phi_{ext}/\varphi_0$')
    #ax[0].set_title(r'$t_{ee}=$1  $\theta=\pi/2$  $g_{c,R}=$0.2')
#    ax[0].legend()

    ax[1].grid(c='lightgrey')
    ax[0].set_ylim([4.9,8])
    ax[1].set_ylim([-20,20])
    ax[2].set_ylim([-2,2])

    ax[1].set_ylabel(r'$g3/2\pi$ (MHz)')
    ax[1].set_xlabel(r'$\phi_{ext}/\varphi_0$')
    ax[2].set_ylabel(r'$Kerr/2\pi$ (MHz)')
    ax[2].set_xlabel(r'$\phi_{ext}/\varphi_0$')
    ax[3].set_ylabel(r'$kappa_phi/2\pi$ (kHz)')
    ax[3].set_xlabel(r'$\phi_{ext}/\varphi_0$')