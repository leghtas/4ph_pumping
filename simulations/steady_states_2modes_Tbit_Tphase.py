# -*- coding: utf-8 -*-
"""
Created on Tue Jan  23 17:16:36 2018

@author: leghtas
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import os
import time as truetime
from matplotlib.colors import LinearSegmentedColormap


cdict = {'red':  ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 1.0, 1.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 0.0, 0.0))
        }
wigner_cm = LinearSegmentedColormap('my_cm', cdict)

def to_pcolor(x, y):
    xf = 2*x[-1]-x[-2]
    x = np.append(x, xf)
    x = x-(x[1]-x[0])/2
    if len(y)>1:
        yf = 2*y[-1]-y[-2]
        y = np.append(y, yf)
        y = y-(y[1]-y[0])/2
    else:
        y = np.append(y, y[0]+1)
        y = y-(y[1]-y[0])/2    
    return (x, y)


plt.close('all')

solve1mode = False
solve2mode = True
solve3mode = False


g2 = 2 * np.pi * 0.5



kappab = 2 * np.pi * 10
kappab_i = 2*np.pi*0.2

kappa2 = 4*g2**2/kappab


Na = 30
Nb = 8
a = qt.destroy(Na)
aI = qt.tensor(a, qt.identity(Nb))
b = qt.destroy(Nb)
Ib = qt.tensor(qt.identity(Na), b)
xvec = np.linspace(-3, 3, 101)

dmax = kappab/5
numd = 21
delta_d_Vec = np.linspace(-dmax, dmax, numd)
#delta_d_Vec = (delta_d_Vec+(delta_d_Vec[1]-delta_d_Vec[0])/2)[:-1]
#numd = numd-1


debug=True
buf_seul=True
if debug:
    numd = 1
    delta_d_Vec = np.array([0])
at = np.zeros((len(delta_d_Vec)), dtype=complex)
nbt = np.zeros((len(delta_d_Vec)), dtype=complex)
bt = np.zeros((len(delta_d_Vec)), dtype=complex)

kappa_phase = 1/10*0
kappa_bit = 1/10*0
kappaa = 1/10
kappaaa = 1/10*0
S21s = []
S21s_buf = []
bts = []
bts_buf = []
kappa_phases = [1/10*0]#, 1/1, 1/10]#, 1/100]

eps_drive = np.exp(1j*np.pi/4) #direct 1ph drive on the memory

for kappa_phase in kappa_phases:
    bins = g2/np.sqrt(kappab)*np.array([3,5.48,10])
    for kk, bIn in enumerate(bins):
        print('%s / %s' % (kk+1, len(bins)))
        alpha0 = (1j*(np.sqrt(kappab)* bIn)/g2)**0.5
        keff = kappa2*np.abs(alpha0)
        print('ka = %.3f, keff = %.3f'%(kappaa, keff))
        print('alpha0 = %.3f'%np.abs(alpha0))
        ket_alpha0 = qt.coherent(Na, alpha0)
        ket_malpha0 = qt.coherent(Na, -alpha0)
        if debug:
            fig_debug, ax_debug = plt.subplots(2,3)
            W2 = qt.wigner(ket_alpha0, xvec, xvec, g=2)
            ax_debug[0,0].pcolor(xvec, xvec, W2, cmap=wigner_cm, vmin=-1, vmax=1)
            ax_debug[0,0].set_aspect('equal')

        phase_flip = ket_alpha0*ket_alpha0.dag()-ket_malpha0*ket_malpha0.dag()
        bit_flip = ket_alpha0*ket_malpha0.dag()+ket_malpha0*ket_alpha0.dag()
        # No drive here
        H2 = (g2*(qt.tensor(a**2, b.dag()) + qt.tensor(a.dag()**2, b)))+ \
             (eps_drive*aI.dag()-np.conj(eps_drive)*aI)/1j
        
#        c_ops2 = [np.sqrt(kappaa*0)*qt.tensor(a, qt.identity(Nb)),
#                  np.sqrt(kappab+kappab_i)*qt.tensor(qt.identity(Na), b),
#                  np.sqrt(kappaphi)*qt.tensor(a.dag()*a, qt.identity(Nb)),
#                  np.sqrt(kappa_phaseflip)*qt.tensor(phase_flip, qt.identity(Nb))]
        c_ops2 = [np.sqrt(kappa_bit)*qt.tensor(bit_flip, qt.identity(Nb)),
                  np.sqrt(kappab+kappab_i)*qt.tensor(qt.identity(Na), b),
                  np.sqrt(kappa_phase)*qt.tensor(phase_flip, qt.identity(Nb)), 
                  np.sqrt(kappaa)*qt.tensor(a, qt.identity(Nb)),
                  np.sqrt(kappaaa)*qt.tensor(a.dag()*a, qt.identity(Nb))]
        
        print(kappa_bit, kappa_phase)
        
        
        for jj, delta_d in enumerate(delta_d_Vec):
            print('%s / %s' % (jj+1, len(delta_d_Vec)))
            Hdet = np.sqrt(kappab)* bIn * qt.tensor(qt.identity(Na), b.dag() - b)/1j + \
                   delta_d * qt.tensor(qt.identity(Na), b.dag()*b) + \
                   delta_d/2 * qt.tensor(a.dag()*a, qt.identity(Nb))
        
            try:
                rho_ss = qt.steadystate(Hdet + H2, c_ops2)# + H2
                success=True
            except Exception:
                print('######FAILED######')
                success=False
            if success:
                nbt[jj] = qt.expect(Ib.dag()*Ib, rho_ss)
                bt[jj] = qt.expect(Ib, rho_ss)
            else:
                nbt[jj] = np.nan
                bt[jj] = np.nan
        print('nbar_b max = %.3f'%(np.nanmax(np.abs(bt)**2)))
        bOut = bt*np.sqrt(kappab)+bIn
        S21 = bOut/bIn
        bts.append(bt.copy())
        S21s.append(S21.copy())

            
        if debug:
            rhoa = rho_ss.ptrace(0)
            weight_p = qt.expect(ket_alpha0*ket_alpha0.dag(), rhoa)
            weight_m = qt.expect(ket_malpha0*ket_malpha0.dag(), rhoa)
            print('weight |0\=%.3f, |1\=%.3f, tot=%.3f'%(weight_p, weight_m, weight_m+weight_p))
#            rhoa_flip = phase_flip*rhoa*phase_flip.dag()
#            rhoa = (rhoa+rhoa_flip)/2
            Wa = qt.wigner(rhoa, xvec, xvec, g=2)
            ax_debug[1,0].pcolor(xvec, xvec, Wa, cmap=wigner_cm, vmin=-1, vmax=1)
            ax_debug[1,0].set_aspect('equal')
            ax_debug[1,0].set_title('SS mem')
            
            rhob = rho_ss.ptrace(1)
            Wb = qt.wigner(rhob, xvec, xvec, g=2)
            ax_debug[1,1].pcolor(xvec, xvec, Wb, cmap=wigner_cm, vmin=-1, vmax=1)
            ax_debug[1,1].set_aspect('equal')
            ax_debug[1,1].set_title('SS buf')
        



        if buf_seul:
            beta0 = 2*np.sqrt(kappab)* bIn/(kappab+kappab_i)
            print('beta0 = %.3f'%np.abs(beta0))
            for jj, delta_d in enumerate(delta_d_Vec):
#                print('%s / %s' % (jj+1, len(delta_d_Vec)))
#                Hdet_buf = np.sqrt(kappab)* bIn * (b.dag() - b)/1j + \
#                       delta_d * b.dag()*b
#                c_ops_buf = [np.sqrt(kappab+kappab_i)*b]     
#                try:
#                    rho_ss_buf = qt.steadystate(Hdet_buf, c_ops_buf)
#                    success=True
#                except Exception:
#                    print('######FAILED###### (buf seul)')
#                    success=False
#                if success:
#                    bt[jj] = qt.expect(b, rho_ss_buf)
#                else:
#                    bt[jj] = np.nan
                beta_comp = -np.sqrt(kappab)* bIn/((kappab+kappab_i)/2+1j*delta_d)
                bt[jj]=beta_comp
                ket_beta0=qt.coherent(Nb, beta_comp)
                rho_ss_buf=ket_beta0*ket_beta0.dag()
                
            bOut = bt*np.sqrt(kappab)+bIn
            S21 = bOut/bIn
            bts_buf.append(bt.copy())
            S21s_buf.append(S21.copy())

            if debug:
                Wb = qt.wigner(rho_ss_buf, xvec, xvec, g=2)
                ax_debug[1,2].pcolor(xvec, xvec, Wb, cmap=wigner_cm, vmin=-1, vmax=1)
                ax_debug[1,2].set_aspect('equal')
                ax_debug[1,2].set_title('SS buf seul')


S21s=np.array(S21s)
S21s= S21s.reshape(len(bins),numd)
S21s_buf=np.array(S21s_buf)
S21s_buf= S21s_buf.reshape(len(bins),numd)
            
fig, ax = plt.subplots(1,3, figsize = (15,5))
ax[0].plot(delta_d_Vec/2/np.pi, np.abs(S21s)[0].T)
if buf_seul:
    buf = np.abs(S21s_buf)[0].T
    ax[0].plot(delta_d_Vec/2/np.pi, buf)
    delta_lim = (1-np.amin(buf))/2
    center_lim = (1+np.amin(buf))/2
ax[0].set_ylim(center_lim-delta_lim*1.1, center_lim+delta_lim*1.1)

ax[1].plot(delta_d_Vec/2/np.pi, np.abs(S21s)[1].T)
if buf_seul:
    buf = np.abs(S21s_buf)[1].T
    ax[1].plot(delta_d_Vec/2/np.pi, buf)
    delta_lim = (1-np.amin(buf))/2
    center_lim = (1+np.amin(buf))/2
ax[1].set_ylim(center_lim-delta_lim*1.1, center_lim+delta_lim*1.1)

ax[2].plot(delta_d_Vec/2/np.pi, np.abs(S21s)[2].T)
if buf_seul:
    buf = np.abs(S21s_buf)[2].T
    ax[2].plot(delta_d_Vec/2/np.pi, buf)
    delta_lim = (1-np.amin(buf))/2
    center_lim = (1+np.amin(buf))/2
ax[2].set_ylim(center_lim-delta_lim*1.1, center_lim+delta_lim*1.1)


figb, axb = plt.subplots(1,1, figsize = (5,5))
axb.plot(delta_d_Vec/2/np.pi, np.abs(bts)[0].T, color='C0')
axb.plot(delta_d_Vec/2/np.pi, np.abs(bts)[1].T, color='C0')
axb.plot(delta_d_Vec/2/np.pi, np.abs(bts)[2].T, color='C0')
if buf_seul:
    axb.plot(delta_d_Vec/2/np.pi, np.abs(bts_buf)[0].T, color='C1')
    axb.plot(delta_d_Vec/2/np.pi, np.abs(bts_buf)[1].T, color='C1')
    axb.plot(delta_d_Vec/2/np.pi, np.abs(bts_buf)[2].T, color='C1')

plt.show()