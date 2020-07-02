#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:23:52 2019

@author: Vil
"""

import numpy as np
import scipy.linalg as sl
import numpy.linalg as nl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

e = 1.6*1e-19
h = 6.63*1e-34
Phi_0 = h/2/e
hbar = h/2/np.pi
phi_0 = hbar/2/e

def houches(N, Ltune, alpha, CA, LA, phi, rr, printing=False):
    
    omega_plasma = 2*np.pi*plasma_list[rr]*1e9
    C0vs2CJ = 1/ratio[rr]
    
    LA = LA*1e-9
    CA = CA*1e-15
#    print(CA, CJ/N)
    LS0 = Ltune/N*1e-9
    LS  = LS0/np.sqrt(1+(alpha*np.tan(phi/2))**2)/np.cos(phi/2)
    LS = np.abs(LS)
    CS = 1/(omega_plasma**2*LS0) 
    C0 = CS*C0vs2CJ #CS=2CJ
#    print(CS/N+N*C0)
    print(CS/2/N, N*C0)
    T = np.zeros((N+1,N+1)) # C
    U = np.zeros((N+1,N+1)) # L^-1
    for i in range(1,N):
        T[i][i] = C0+2*CS
        T[i][i+1] = -CS
        U[i][i] = 2/LS
        U[i][i+1] = -1/LS
    T[0][0] = CA
    T[1][1] += -CS
    T[-1][-1] = C0+2*CS
    U[0][0] = 1/LA
    U[1][1] += 1/LA - 1/LS
    U[0][1] = -1/LA
    U[-1][-1] = 2/LS
    T = T + T.transpose() - np.diag(np.diag(T))
    U = U + U.transpose() - np.diag(np.diag(U))
    
    if printing:
        print(T)
        print(U)
        
    w, v = nl.eigh(T)
    sqrT = sl.sqrtm(np.diag(w))
    U1 = np.dot(np.dot(sl.inv(v), U), v)
    U2 = np.dot(np.dot(sl.inv(sqrT), U1), sl.inv(sqrT))
    w, v = nl.eigh(U2)

#    does not work for some reason
#    Cm12 = (np.dot(np.dot(sl.inv(v), sl.inv(sqrT)), v))
#    Cm12 = sl.inv(sqrT)
#    eta = 0
#    for jj in range(len(Cm12)):
#        et = Cm12[jj][0]*v[0][k]
#        for ii in range(1,len(Cm12[jj])):
#            et += (Cm12[jj][ii] - Cm12[jj][ii-1])*v[ii][k]
#        eta += et**4
#    eta = eta*CS**2
#    kerr = 2*hbar*np.pi**4*(phi_0**2/LS)*eta/(phi_0**4)/(CS**2)/w[k]
    
    kerr = []
    for k in range(N+1):
        kerr.append((1/2+1/8)*hbar**2*w[k]/2/N/(phi_0**2/LS) / hbar )#missing hbar in the paper !
    kerr = np.array(kerr)
    
    return np.sqrt(w)/2/np.pi/1e9, kerr/2/np.pi/1e6
    
def update(val):
    N = int(s_N.val)
    Ltune = s_Ltune.val
    alpha = s_alpha.val
    CA = s_CA.val
    LA = s_LA.val

    fmatrix, kerrmatrix = [], []
    for phi in phi_list:
        fmat, kerrmat = houches(N, Ltune, alpha, CA, LA, phi, rr, printing=False)
        fmatrix.append(fmat)
        kerrmatrix.append(kerrmat)
    fmatrix = np.array(fmatrix).transpose()
    kerrmatrix = np.array(kerrmatrix).transpose()
    lmat0.set_ydata(fmatrix[0])
    lkerr0.set_ydata(kerrmatrix[0])
    lmat1.set_ydata(fmatrix[1])
    lkerr1.set_ydata(kerrmatrix[1])
    lmat2.set_ydata(fmatrix[2])
    lkerr2.set_ydata(kerrmatrix[2])
    
    fig.canvas.draw_idle()
    
def reset(event):
    s_N.reset()
    s_Ltune.reset()
    s_alpha.reset()
    s_LA.reset()
    s_CA.reset()

''''''''''''''''''''
plt.close('all')

plasma_list = [16, 20, 24]
ratio = [91, 53, 35]

phi_list = np.linspace(-np.pi, 3*np.pi, 201)
N_list = list(range(5,201))
param = [20, 3.75, 0.25, 25, 1, 0, 0] #N, Ltune, alpha, CA, LA, phi, rr
#param = [40, 13, 0.33, 0.1,3, 0, 0] #N, Ltune, alpha, CA, LA, phi, rr
    
if 1:
    fig, ax = plt.subplots(1,2,figsize=(10,8))
    plt.subplots_adjust(left=0.25, bottom=0.35)
    
    N0, Ltune0, alpha0, CA0, LA0, _, rr = param #N, Ltune, alpha, CA, LA, phi, rr
    
    fmatrix, kerrmatrix = [], []
    for phi in phi_list:
        fmat, kerrmat = houches(N0, Ltune0, alpha0, CA0, LA0, phi, rr) #N, Ltune, alpha, CA, LA, phi, rr, printing=False
        fmatrix.append(fmat)
        kerrmatrix.append(kerrmat)
    fmatrix = np.array(fmatrix).transpose()
    kerrmatrix = np.array(kerrmatrix).transpose()
    lmat0, = ax[0].plot(phi_list, fmatrix[0])
    lkerr0, = ax[1].plot(phi_list, kerrmatrix[0])
    lmat1, = ax[0].plot(phi_list, fmatrix[1])
    lkerr1, = ax[1].plot(phi_list, kerrmatrix[1])
    lmat2, = ax[0].plot(phi_list, fmatrix[2])
    lkerr2, = ax[1].plot(phi_list, kerrmatrix[2])
    
    axcolor = 'white'
    ax_N = plt.axes([0.25, 0, 0.65, 0.03], facecolor=axcolor)
    ax_Ltune = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    ax_alpha = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_CA = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_LA = plt.axes([0.25, 0.2, 0.25, 0.03], facecolor=axcolor)
    s_N = Slider(ax_N, r'$N$', 1, 20, valinit=N0)
    s_Ltune = Slider(ax_Ltune, r'$L_{tune}$ (nH)', 0.1, 8, valinit=Ltune0)
    s_alpha = Slider(ax_alpha, r'$\alpha$', 0.1, 0.9, valinit=alpha0)
    s_CA = Slider(ax_CA, r'$C_A (fF)$', 0.1, 200, valinit=CA0)
    s_LA = Slider(ax_LA, r'$L_A$ (nH)', 0.1, 15, valinit=LA0)
    
    s_N.on_changed(update)
    s_Ltune.on_changed(update)
    s_alpha.on_changed(update)
    s_CA.on_changed(update)
    s_LA.on_changed(update)
    
    resetax = plt.axes([0.05, 0.15, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    button.on_clicked(reset)
    
    ax[0].grid(c='lightgrey')
#    ax[0].set_ylim([0,15])
    ax[0].set_ylabel(r'$\omega/2\pi$ (GHz)')
    ax[0].set_xlabel(r'$\phi_{ext}/\varphi_0$')
    #ax[0].set_title(r'$t_{ee}=$1  $\theta=\pi/2$  $g_{c,R}=$0.2')
#    ax[0].legend()
    
    ax[1].grid(c='lightgrey')
    ax[1].set_ylim([0,5])
    ax[1].set_ylabel(r'$\chi/2\pi$ (MHz)')
    ax[1].set_xlabel(r'$\phi_{ext}/\varphi_0$')
