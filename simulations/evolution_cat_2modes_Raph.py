# -*- coding: utf-8 -*-
"""
Created on Tue Jan  23 17:16:36 2018

@author: leghtas
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from matplotlib.colors import LinearSegmentedColormap
import os

def to_pcolor(x, y):
    xf = 2*x[-1]-x[-2]
    yf = 2*y[-1]-y[-2]
    x = np.append(x, xf)
    y = np.append(y, yf)
    x = x-(x[1]-x[0])/2
    y = y-(y[1]-y[0])/2
    return (x, y)

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


plt.close('all')
plt.ion()

solve1mode = False
solve2mode = True
solve3mode = False

nbar0 = 0.1
alpha0_pump = np.sqrt(nbar0)*np.exp(1j*np.pi/2)
g2 = 2 * np.pi * 1
Kerr = 2 * np.pi * 0.05*0

kappaa = 1. / 0.4*0
kappab = 2 * np.pi * 20.
chiab = 2*np.pi *0.1*0
kappaphi = 0*kappaa/2 #1 * 2 * np.pi * 0.2
drive = 0

delta_p = 2*np.pi*0
#delta_d = -delta_p
delta_d = 2*np.pi*0

# 1 mode
kappa2 = 4*g2**2/kappab  # 2*np.pi MHz
print('kappa_2ph = %.3f MHz, T_2ph = %.3f µs'%(kappa2/2/np.pi, 1/kappa2))
# 3 mode
chi = 2 * np.pi * 1.
nth = 0.2
T1 = 2

Na = 25
Nb = 2
a = qt.destroy(Na)
b = qt.destroy(Nb)


#
H1 = -Kerr * a.dag()**2*a**2 + drive*(a.dag()-a)/1j
H2 = (qt.tensor(H1, qt.identity(Nb)) +
      g2*qt.tensor((a**2-alpha0_pump**2), b.dag()) +
          g2*qt.tensor((a.dag()**2-alpha0_pump.conjugate()**2), b) -
      chiab*(qt.tensor(a.dag()*a, b.dag()*b)))
H2det = ((delta_p+delta_d)/2 * qt.tensor(a.dag()*a, qt.identity(Nb)) +
        delta_d * qt.tensor(qt.identity(Na), b.dag()*b))
H3 = qt.tensor(H2, qt.identity(2)) + chi * qt.tensor(a.dag()*a, qt.identity(Nb),
               (qt.sigmaz()+qt.identity(2))/2)

psi01 = qt.coherent(Na, 0*alpha0_pump)
ketalpha0 = qt.coherent(Na, alpha0_pump)
ketmalpha0 = qt.coherent(Na, -alpha0_pump)

c_ops1 = [np.sqrt(kappaa)*a,
          np.sqrt(kappa2)*(a**2-alpha0_pump**2),
          np.sqrt(kappaphi)*(a.dag()*a)]
c_ops2 = [np.sqrt(kappaa)*qt.tensor(a, qt.identity(Nb)),
          np.sqrt(kappab)*qt.tensor(qt.identity(Na), b),
          np.sqrt(kappaphi)*qt.tensor(a.dag()*a, qt.identity(Nb))]
c_ops3 = [np.sqrt(kappaa)*qt.tensor(a, qt.identity(Nb), qt.identity(2)),
          np.sqrt(kappab)*qt.tensor(qt.identity(Na), b, qt.identity(2)),
          np.sqrt(kappaphi)*qt.tensor(a.dag()*a, qt.identity(Nb), qt.identity(2)),
          np.sqrt(nth/T1)*qt.tensor(qt.identity(Na), qt.identity(Nb), qt.sigmap()),
          np.sqrt((1+nth)/T1)*qt.tensor(qt.identity(Na), qt.identity(Nb), qt.sigmam())]

alpha0_prep = np.sqrt(0.1)*np.exp(1j*np.pi/2)


Nwig = 101
xvec = np.linspace(-np.abs(alpha0_pump)-1, np.abs(alpha0_pump)+1, Nwig)
xarr, yarr = np.meshgrid(xvec, xvec)
alphaarr = xarr+1j*yarr
dx = xvec[1]-xvec[0]
#fig0, ax0 = plt.subplots()
#rho_ss_3 = qt.steadystate(H3, c_ops3)
#rhoa3 = rho_ss_3.ptrace(0)
#W3 = qt.wigner(rhoa3, xvec, xvec)
#ax0.pcolor(xvec, xvec, W3)
#ax0.axis('equal')



psi01_0 = qt.coherent(Na, alpha0_prep)
psi01_1 = qt.coherent(Na, -alpha0_prep)

overlap = np.abs((psi01_0.dag()*psi01_1.full())[0,0])
print('###### OVERLAP ######')
print('Overlap = %.3f'%overlap)
psi01_plus = (psi01_0+psi01_1)/(psi01_0+psi01_1).norm()
psi01_minus = (psi01_0-psi01_1)/(psi01_0-psi01_1).norm()
overlap2 = np.abs((psi01_plus.dag()*psi01_minus.full())[0,0])
print('Overlap_cat = %.3f'%overlap2)

psi01_0 = (psi01_plus+psi01_minus)/2**0.5
psi01_1 = (psi01_plus-psi01_minus)/2**0.5

psi02 = qt.tensor(psi01_0, qt.basis(Nb, 0))
psi02_bis = qt.tensor(psi01_1, qt.basis(Nb, 0))
psi03 = qt.tensor(psi01, qt.basis(Nb, 0), qt.basis(2,1))
rho03 = qt.tensor(psi01*psi01.dag(), qt.basis(Nb, 0)*qt.basis(Nb, 0).dag(),
                  qt.projection(2, 1, 1)*(1-nth)+qt.projection(2, 0, 0)*nth)

e1 = 2*np.pi*0.0
omega_drive = 2*np.pi*0.0


H1_drive = (e1*a.dag()-np.conj(e1)*a)/1j
H1_drive1 = qt.tensor(e1*a.dag()/1j,qt.identity(Nb))
H1_drive2 = qt.tensor(-np.conj(e1)*a/1j,qt.identity(Nb))
def H1_drive1_coeff(t, args):
    return np.exp(1j*args['omega_drive']*t)
def H1_drive2_coeff(t, args):
    return np.exp(-1j*args['omega_drive']*t)


parity_operator = (1j*np.pi*a.dag()*a).expm()

if 1==1:
    #T = 5/kappa2
    T = 10
    dt = T/100.
    tlist = np.linspace(0, T, T/dt+1)
    fig, ax = plt.subplots()
    ax.plot(tlist, np.real(H1_drive1_coeff(tlist, {'omega_drive': omega_drive})))
    ax.set_title('drive')
    if solve1mode:
        result = qt.mesolve(H1, psi01, tlist, c_ops1, [])
        at = []
        alphat = []
        malphat = []
        for ii, t in enumerate(tlist):
            at.append(qt.expect(a**2, result.states[ii]))
            alphat.append(qt.expect(result.states[ii], ketalpha0))
            malphat.append(qt.expect(result.states[ii], ketmalpha0))
    if solve2mode:
        fock0 = qt.basis(Na, 0)
        result2 = qt.mesolve([H2+0*H2det,[H1_drive1, H1_drive1_coeff], [H1_drive2, H1_drive2_coeff]], psi02, tlist, c_ops2, [],  args={'omega_drive': omega_drive})
        result2_bis = qt.mesolve([H2+0*H2det,[H1_drive1, H1_drive1_coeff], [H1_drive2, H1_drive2_coeff]], psi02_bis, tlist, c_ops2, [],  args={'omega_drive': omega_drive})
        energy_0 = qt.expect(H1_drive, ketalpha0)
        energy_1 = qt.expect(H1_drive, ketmalpha0)
        print('energy |0\ = %.3f'%energy_0)
        print('energy |1\ = %.3f'%energy_1)
        pop0 = []
        at2 = []
        bt2 = []
        alphat2 = []
        malphat2 = []
        popup = []
        popdown = []
        W2s = []
        W2s_bis = []
        parity = []
        nbara = []
        for ii, t in enumerate(tlist):
            at2.append(qt.expect(qt.tensor(a**2, qt.identity(Nb)),
                                 result2.states[ii]))
            bt2.append(qt.expect(qt.tensor(qt.identity(Na), b**2),
                                 result2.states[ii]))
            rhoa = result2.states[ii].ptrace(0)
            rhoa_bis = result2_bis.states[ii].ptrace(0)
            W2 = qt.wigner(rhoa, xvec, xvec, g=2)
            W2_bis = qt.wigner(rhoa_bis, xvec, xvec, g=2)
            W2s.append(W2)
            W2s_bis.append(W2_bis)
            dist = np.sum(W2, axis=1)*dx
    
            popup.append(np.sum(dist[0:int(Nwig/2)+1])*dx)
            popdown.append(np.sum(dist[int(Nwig/2)+1:-1])*dx)
            alphat2.append(qt.expect(rhoa, ketalpha0))
            malphat2.append(qt.expect(rhoa, ketmalpha0))
            pop0.append(qt.expect(rhoa, fock0))
            parity.append(qt.expect(rhoa, parity_operator))
            nbara.append(qt.expect(rhoa, a.dag()*a))
    if solve3mode:
        result3 = qt.mesolve(H3, rho03, tlist, c_ops3, [])
        at3 = []
        bt3 = []
        alphat3 = []
        malphat3 = []
        for ii, t in enumerate(tlist):
            at3.append(qt.expect(qt.tensor(a**2, qt.identity(Nb), qt.identity(2)),
                                 result3.states[ii]))
            bt3.append(qt.expect(qt.tensor(qt.identity(Na), b**2, qt.identity(2)),
                                 result3.states[ii]))
            rhoa = result3.states[ii].ptrace(0)
            alphat3.append(qt.expect(rhoa, ketalpha0))
            malphat3.append(qt.expect(rhoa, ketmalpha0))
    
    
    fig, ax = plt.subplots()
    if solve1mode:
        ax.plot(tlist, np.real(at), label='real')
        ax.plot(tlist, np.imag(at), label='imag')
    if solve2mode:
        ax.plot(tlist, pop0, label='pop0')
        ax.plot(tlist, np.real(at2), label='real a 2')
        ax.plot(tlist, np.imag(at2), label='imag a 2')
        ax.plot(tlist, np.real(bt2), label='real b 2')
        ax.plot(tlist, np.imag(bt2), label='imag b 2')
    if solve3mode:
        ax.plot(tlist, np.real(at3), label='3')
        ax.plot(tlist, np.real(bt3), label='3')
    ax.set_xlabel('Time (us)')
    ax.legend()
    
    fig2, ax2 = plt.subplots()
    if solve1mode:
        ax2.plot(tlist, np.real(alphat), label='one mode +alpha')
        ax2.plot(tlist, np.real(malphat), label='one mode -alpha')
    if solve2mode:
        ax2.plot(tlist, np.real(alphat2), label='two mode +alpha')
        ax2.plot(tlist, np.real(malphat2), label='two mode -alpha')
        ax2.plot(tlist, popup, label='two mode popup')
        ax2.plot(tlist, popdown, label='two mode popdown')
    if solve3mode:
        ax2.plot(tlist, np.real(alphat3), label='3 mode +alpha')
        ax2.plot(tlist, np.real(malphat3), label='3 mode -alpha')
        pass
    ax2.set_xlabel('Time (us)')
    ax2.set_ylabel('proj on alpha')
    ax2.legend()
    fig3, ax3 = plt.subplots(2,5, figsize=(12,6))
    fig4, ax4 = plt.subplots(2,5, figsize=(12,6))
    T_disp = T
    dt_disp = T_disp/10

    tdisplay = np.linspace(0,T_disp, T_disp/dt_disp+1)[:-1]

    for ii, tdisp in enumerate(tdisplay):
        index_T = np.argmin(np.abs(tlist-tdisp))
        ax3[ii//5,ii%5].pcolor(xvec, xvec, np.pi/2*(W2s[index_T]+0*W2s_bis[index_T]), cmap=wigner_cm, vmin=-1, vmax=1)
        ax3[ii//5,ii%5].set_aspect('equal')
        ax3[ii//5,ii%5].set_title('t = %.1f µs'%tdisp)
        ax4[ii//5,ii%5].pcolor(xvec, xvec, np.pi/2*(0*W2s[index_T]+W2s_bis[index_T]), cmap=wigner_cm, vmin=-1, vmax=1)
        ax4[ii//5,ii%5].set_aspect('equal')
        ax4[ii//5,ii%5].set_title('t = %.1f µs'%tdisp)
        if ii==len(tdisplay)-1:
            disp = (W2s[index_T]*alphaarr).sum()*dx**2
            print(disp)
            print(drive/np.abs(alpha0_pump)**2/kappa2/2)
            
    folder = os.getcwd()
    image_dir = os.path.join(folder,"simus")
    if not os.path.isdir(image_dir): os.makedirs(image_dir)
    
    image_name = 'g2o2pi_%s_kappaao2pi_%s_nbar0_%s_kappaphio2pi_%s' % \
                    (g2/2/np.pi, kappaa/2/np.pi, np.abs(alpha0_pump)**2, kappaphi)
    fig3.suptitle(image_name)
    fig2.suptitle(image_name)

    fig3.savefig(os.path.join(image_dir, image_name+'_wig.png'))
    fig2.savefig(os.path.join(image_dir, image_name+'.png'))
    plt.show()
    fign, axn = plt.subplots(1, 2)
    axn[0].plot(tlist, parity)
    axn[0].set_xlabel('time')
    axn[0].set_ylabel('parity')
    
    axn[1].plot(tlist, nbara)
    axn[1].set_xlabel('time')
    axn[1].set_ylabel('nbar_a')

if 1==0:
    fig3, ax3 = plt.subplots(2,2, figsize=(8,8))
    if solve1mode:
        W = qt.wigner(result.states[-1], xvec, xvec)
        ax3[0,0].pcolor(xvec, xvec, W)
        ax3[0,0].set_title('1mode')
    if solve2mode:
        W2 = qt.wigner(qt.ptrace(result2.states[-1], 0), xvec, xvec)
        ax3[1,0].pcolor(xvec, xvec, W2)
        ax3[1,0].set_title('2modes')
    if solve3mode:
        W3 = qt.wigner(qt.ptrace(result3.states[-1], 0), xvec, xvec)
        ax3[1,1].pcolor(xvec, xvec, W3)
        ax3[1,1].set_title('3modes')
    ax3[0,0].set_aspect(1)
    ax3[1,0].set_aspect(1)
    ax3[1,1].set_aspect(1)
    
if 1==0:
    dmax = 0.01 * kappab
    pmax = 0.01 * kappab
    numd = 11
    nump = 11
    delta_p_Vec = np.linspace(-pmax, pmax, nump)
    delta_d_Vec = np.linspace(-dmax, dmax, numd)
    parity = np.zeros((len(delta_p_Vec), len(delta_d_Vec)), dtype=float)
    
    parity_operator = (1j*np.pi*a.dag()*a).expm()
    parity_operator = qt.tensor(parity_operator, qt.identity(Nb))
    
    indices_d = [0, (numd-1)/2, numd-1]
    indices_p = [0, (nump-1)/2, nump-1]
    
    rhoss = []
    
    for jj, delta_d in enumerate(delta_d_Vec):
        print('%s / %s' % (jj+1, len(delta_d_Vec)))
        for ii, delta_p in enumerate(delta_p_Vec):
            H2det = ((delta_p+delta_d)/2 * qt.tensor(a.dag()*a, qt.identity(Nb)) +
                    delta_d * qt.tensor(qt.identity(Na), b.dag()*b))
            try:
                rho_ss = qt.steadystate(H2det + H2, c_ops2)
            except Exception:
                print(f'failed jj={jj},ii={ii}')
            parity[ii, jj] = qt.expect(parity_operator, rho_ss)
            if (ii in indices_p) and (jj in indices_d):
                _ii, _jj = (indices_p.index(ii), indices_d.index(jj))
                rhoss.append(rho_ss)
            
    rhoss = np.array(rhoss).reshape((3,3))
    
    fig, ax = plt.subplots()
    _delta_d_Vec, _delta_p_Vec = to_pcolor(delta_d_Vec, delta_p_Vec)
    ax.pcolor(_delta_d_Vec/2/np.pi, _delta_p_Vec/2/np.pi, parity, vmin=-1, vmax=1)
    ax.set_xlabel('delta shine')
    ax.set_ylabel('delta pump')
    
    fig2, ax2 = plt.subplots(3,3)
    for jj in range(3):
        for ii in range(3):
            ax.plot(delta_d_Vec[int(indices_d[jj])]/np.pi/2, delta_p_Vec[int(indices_p[ii])]/np.pi/2, 'x', color='k')
            rho_ss = rhoss[jj,ii]
            W2 = qt.wigner(qt.ptrace(rho_ss, 0), xvec, xvec)
            ax2[2-ii, jj].pcolor(xvec, xvec, W2)

