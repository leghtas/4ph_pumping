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


bIn = 1
g2 = 2 * np.pi * 5
Kerr = 0
Kerrb = 2*np.pi*0

kappaa = 1. / 100
kappab = 2 * np.pi * 20
kappab_i = 2*np.pi*0.0
kappaphi =  0* 2 * np.pi * 0.1

kappa2 = 4*g2**2/kappab

# 3 mode
chi = 2 * np.pi * 3.
nth = 0.1
T1 = 15

Na = 20
Nb = 3
a = qt.destroy(Na)
aI = qt.tensor(a, qt.identity(Nb))
b = qt.destroy(Nb)
Ib = qt.tensor(qt.identity(Na), b)

H1 = -Kerr/2* a.dag()**2*a**2
H1b = -Kerrb/2 * b.dag()**2*b**2

dmax = 1 * kappab
pmax = 0.0 * kappab
#pmax = 300/40
numd = 101
nump = 1
delta_p_Vec = np.linspace(-pmax, pmax, nump)
delta_d_Vec = np.linspace(-dmax, dmax, numd)
at = np.zeros((len(delta_p_Vec), len(delta_d_Vec)), dtype=complex)
bt = np.zeros((len(delta_p_Vec), len(delta_d_Vec)), dtype=complex)

bins = np.logspace(-2,2,5)
S21s = []
for bIn in bins:
    # No drive here
    H2 = (0*qt.tensor(H1, qt.identity(Nb)) + 0*qt.tensor(qt.identity(Na), H1b) + 
          g2*(qt.tensor(a**2, b.dag()) + qt.tensor(a.dag()**2, b)))
    
    c_ops2 = [np.sqrt(kappaa)*qt.tensor(a, qt.identity(Nb)),
              np.sqrt(kappab+kappab_i)*qt.tensor(qt.identity(Na), b),
              np.sqrt(kappaphi)*qt.tensor(a.dag()*a, qt.identity(Nb))]
    
    
    for jj, delta_d in enumerate(delta_d_Vec):
        print('%s / %s' % (jj+1, len(delta_d_Vec)))
        for kk, delta_p in enumerate(delta_p_Vec):
            Hdet = np.sqrt(kappab)* bIn * qt.tensor(qt.identity(Na), b.dag() - b)/1j + \
                   delta_d * qt.tensor(qt.identity(Na), b.dag()*b) + \
                   (delta_p+delta_d)/2 * qt.tensor(a.dag()*a, qt.identity(Nb))
        
            try:
                rho_ss = qt.steadystate(Hdet + H2, c_ops2)# + H2
            except Exception:
                print(f'failed jj={jj},kk={kk}')
            at[kk, jj] = qt.expect(aI.dag()*aI, rho_ss)
            bt[kk, jj] = qt.expect(Ib, rho_ss)
    
    bOut = bt*np.sqrt(kappab)+bIn
    S21 = bOut/bIn
S21s=np.array(S21s)

#bt = bt + 1j*(g2/kappab)*alpha1**2

fig, ax = plt.subplots(2,3)
_delta_d_Vec, _delta_p_Vec = to_pcolor(delta_d_Vec, delta_p_Vec)
plt_abs = np.abs(S21)
plt_angle = np.unwrap(np.angle(S21))
print((np.max(plt_angle)-np.min(plt_angle))*180/np.pi)
abs_mean = np.mean(plt_abs)
abs_std = np.std(plt_abs)
ax[0,0].pcolor(_delta_d_Vec/2/np.pi, _delta_p_Vec/2/np.pi, plt_abs, vmin = 0)
ax[1,0].pcolor(_delta_d_Vec/2/np.pi, _delta_p_Vec/2/np.pi, plt_angle)
ax[0,0].plot([2*g2*2**(1/2)/2/2/np.pi, 2*g2*2**(1/2)/2/2/np.pi], [_delta_p_Vec[0]/2/np.pi, _delta_p_Vec[-1]/2/np.pi])
ax[0,0].plot([-2*g2*2**(1/2)/2/2/np.pi, -2*g2*2**(1/2)/2/2/np.pi], [_delta_p_Vec[0]/2/np.pi, _delta_p_Vec[-1]/2/np.pi])
ax[0,0].plot([_delta_d_Vec[0]/2/np.pi, _delta_d_Vec[-1]/2/np.pi], [0,0])
ax[0,0].axis('tight')
ax[1,0].axis('tight')

index_middle_pump = np.argmin(np.abs(delta_p_Vec))
ax[0,1].plot(delta_d_Vec/2/np.pi, plt_abs[index_middle_pump])
ax[0,1].set_ylim(top=1)
ax[1,1].plot(delta_d_Vec/2/np.pi, plt_angle[index_middle_pump])

plt_abs = np.abs(at)
print(np.max(plt_abs))
plt_angle = np.angle(at)
abs_mean = np.mean(plt_abs)
abs_std = np.std(plt_abs)
ax[0,2].pcolor(_delta_d_Vec/2/np.pi, _delta_p_Vec/2/np.pi, plt_abs)
ax[1,2].pcolor(_delta_d_Vec/2/np.pi, _delta_p_Vec/2/np.pi, plt_angle)
#ax[0,1].plot([2*g2*2**(1/2)/2/2/np.pi, 2*g2*2**(1/2)/2/2/np.pi], [_delta_p_Vec[0]/2/np.pi, _delta_p_Vec[-1]/2/np.pi])
#ax[0,1].plot([-2*g2*2**(1/2)/2/2/np.pi, -2*g2*2**(1/2)/2/2/np.pi], [_delta_p_Vec[0]/2/np.pi, _delta_p_Vec[-1]/2/np.pi])
#ax[0,1].plot([_delta_d_Vec[0]/2/np.pi, _delta_d_Vec[-1]/2/np.pi], [0,0])
ax[0,2].axis('tight')
ax[1,2].axis('tight')
plt.show()

image_dir = r".\images\\"
if not os.path.isdir(image_dir):
    os.makedirs(image_dir)
image_name = 'drivememNa_%s_Nb_%s_g2o2pi_%s_nbar0_%s_dmax_%s_numd_%s_pmax_%s_nump_%s' % \
            (Na, Nb, g2/2/np.pi, np.abs(alpha0)**2, dmax, numd, pmax, nump)
plt.savefig(image_dir + image_name + str(truetime.time())+'.png')

bt = bt/np.abs(1j*(g2/kappab)*alpha1**2)
bmax = 1.1
fig, ax = plt.subplots()
ax.plot(np.real(bt[int((nump-1)/2)]), np.imag(bt[int((nump-1)/2)]))
ax.plot(0,0,'x')
ax.set_xlim([-bmax, bmax])
ax.set_ylim([-bmax, bmax])
ax.axis('equal')
