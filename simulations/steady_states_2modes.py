# -*- coding: utf-8 -*-
"""
Created on Tue Jan  23 17:16:36 2018

@author: leghtas
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import os


def to_pcolor(x, y):
    xf = 2*x[-1]-x[-2]
    yf = 2*y[-1]-y[-2]
    x = np.append(x, xf)
    y = np.append(y, yf)
    x = x-(x[1]-x[0])/2
    y = y-(y[1]-y[0])/2
    return (x, y)


plt.close('all')

solve1mode = False
solve2mode = True
solve3mode = False

alpha0 = 0*np.sqrt(0.001)*np.sqrt(2)*np.exp(1j*np.pi/2)
alpha1 = np.sqrt(0.001*10)*np.sqrt(2)*np.exp(1j*np.pi/2)
g2 = 2 * np.pi * 1 * 10
Kerr = 0

kappaa = 1. / 2.*10
kappab = 2 * np.pi * 3.
kappaphi = 0 * 2 * np.pi * 0.1


# 3 mode
chi = 2 * np.pi * 3.
nth = 0.1
T1 = 15

Na = 30
Nb = 2
a = qt.destroy(Na)
aI = qt.tensor(a, qt.identity(Nb))
b = qt.destroy(Nb)
Ib = qt.tensor(qt.identity(Na), b)

H1 = -Kerr * a.dag()**2*a**2
H2 = (qt.tensor(H1, qt.identity(Nb)) +
      g2*(qt.tensor((a**2-alpha0**2), b.dag()) +
          qt.tensor((a.dag()**2-alpha0.conjugate()**2), b)))

c_ops2 = [np.sqrt(kappaa)*qt.tensor(a, qt.identity(Nb)),
          np.sqrt(kappab)*qt.tensor(qt.identity(Na), b),
          np.sqrt(kappaphi)*qt.tensor(a.dag()*a, qt.identity(Nb))]

dmax = 300/40/5*10
pmax = 1500
#pmax = 300/40
numd = 31
nump = 31
delta_p_Vec = np.linspace(-pmax, pmax, nump)
delta_d_Vec = np.linspace(-dmax, dmax, numd)
at = np.zeros((len(delta_p_Vec), len(delta_d_Vec)), dtype=complex)
bt = np.zeros((len(delta_p_Vec), len(delta_d_Vec)), dtype=complex)

for jj, delta_d in enumerate(delta_d_Vec):
    print('%s / %s' % (jj, len(delta_d_Vec)))
    for kk, delta_p in enumerate(delta_p_Vec):
        Hdet = (delta_p+delta_d)/2 * qt.tensor(a.dag()*a, qt.identity(Nb)) + \
            delta_d * qt.tensor(qt.identity(Na), b.dag()*b)
        Hdet2 = delta_d * qt.tensor(a.dag()*a, qt.identity(Nb)) + \
            (2*delta_d-delta_p) * qt.tensor(qt.identity(Na), b.dag()*b) +\
            g2*alpha1**2 * qt.tensor((a.dag() + a), qt.identity(Nb))
        try:
            rho_ss = qt.steadystate(Hdet2 + H2, c_ops2)
        except Exception:
            print(f'failed jj={jj},kk={kk}')
        at[kk, jj] = qt.expect(aI, rho_ss)
        bt[kk, jj] = qt.expect(Ib, rho_ss)

fig, ax = plt.subplots(2)
_delta_d_Vec, _delta_p_Vec = to_pcolor(delta_d_Vec, delta_p_Vec)
plt_abs = np.abs(at)
plt_angle = np.angle(at)
abs_mean = np.mean(plt_abs)
abs_std = np.std(plt_abs)
ax[0].pcolor(_delta_d_Vec/2/np.pi, _delta_p_Vec/2/np.pi, plt_abs)
ax[1].pcolor(_delta_d_Vec/2/np.pi, _delta_p_Vec/2/np.pi, plt_angle)
#ax[0].plot([2*g2*2**(1/2)/2/2/np.pi, 2*g2*2**(1/2)/2/2/np.pi], [_delta_p_Vec[0]/2/np.pi, _delta_p_Vec[-1]/2/np.pi])
#ax[0].plot([-2*g2*2**(1/2)/2/2/np.pi, -2*g2*2**(1/2)/2/2/np.pi], [_delta_p_Vec[0]/2/np.pi, _delta_p_Vec[-1]/2/np.pi])
#ax[0].plot([_delta_d_Vec[0]/2/np.pi, _delta_d_Vec[-1]/2/np.pi], [0,0])
ax[0].axis('tight')
ax[1].axis('tight')
plt.show()

image_dir = r".\images\\"
if not os.path.isdir(image_dir):
    os.makedirs(image_dir)
image_name = 'drivememNa_%s_Nb_%s_g2o2pi_%s_nbar0_%s_dmax_%s_numd_%s_pmax_%s_nump_%s' % \
            (Na, Nb, g2/2/np.pi, np.abs(alpha0)**2, dmax, numd, pmax, nump)
plt.savefig(image_dir + image_name + '.png')
