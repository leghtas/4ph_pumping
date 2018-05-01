# -*- coding: utf-8 -*-
"""
Created on Tue Jan  23 17:16:36 2018

@author: leghtas
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

plt.close('all')
plt.ion()

solve1mode = False
solve2mode = True
solve3mode = False

alpha0 = np.sqrt(3)*np.exp(0*1j*np.pi/2)
g2 = 2 * np.pi * 5.
Kerr = 2 * np.pi * 0.18

kappaa = 1. / 2.
kappab = 2 * np.pi * 3.
kappaphi = 1 * 2 * np.pi * 0

# 1 mode
kappa2 = 4*g2**2/kappab  # MHz

# 3 mode
chi = 2 * np.pi * 3.
nth = 0.03
T1 = 15

Na = 20
Nb = 10
a = qt.destroy(Na)
b = qt.destroy(Nb)
Ia = qt.identity(Na)
Ib = qt.identity(Nb)


def projg(rhoabq):
    bra_g = qt.basis(2, 0).dag()
    ket_g = qt.basis(2, 0)
    rhoab = qt.tensor(Ia, Ib, bra_g) * rhoabq * qt.tensor(Ia, Ib, ket_g)
    rhoab = rhoab/rhoab.tr()
    return rhoab

H1 = - (Kerr/2) * a.dag()**2*a**2
H2 = (qt.tensor(H1, qt.identity(Nb)) +
      g2*(qt.tensor((a**2-alpha0**2), b.dag()) +
          qt.tensor((a.dag()**2-alpha0.conjugate()**2), b)) -
      chiab*(qt.tensor(a.dag()*a, b.dag()*b)))
H3 = qt.tensor(H2, qt.identity(2)) + chi * qt.tensor(a.dag()*a, qt.identity(Nb),
               (qt.sigmaz()+qt.identity(2))/2)

psi01 = qt.coherent(Na, alpha0)
ketalpha0 = qt.coherent(Na, alpha0)
ketmalpha0 = qt.coherent(Na, -alpha0)

c_ops1 = [np.sqrt(kappaa)*a,
          np.sqrt(kappa2)*(a**2-alpha0**2),
          np.sqrt(kappaphi)*(a.dag()*a)]
c_ops2 = [np.sqrt(kappaa)*qt.tensor(a, qt.identity(Nb)),
          np.sqrt(kappab)*qt.tensor(qt.identity(Na), b),
          np.sqrt(kappaphi)*qt.tensor(a.dag()*a, qt.identity(Nb))]
c_ops3 = [np.sqrt(kappaa)*qt.tensor(a, qt.identity(Nb), qt.identity(2)),
          np.sqrt(kappab)*qt.tensor(qt.identity(Na), b, qt.identity(2)),
          np.sqrt(kappaphi)*qt.tensor(a.dag()*a, qt.identity(Nb), qt.identity(2)),
          np.sqrt(nth/T1)*qt.tensor(qt.identity(Na), qt.identity(Nb), qt.sigmap()),
          np.sqrt((1+nth)/T1)*qt.tensor(qt.identity(Na), qt.identity(Nb), qt.sigmam())]

#
Nwig = 101
xvec = np.linspace(-3*np.abs(alpha0), 3*np.abs(alpha0), Nwig)
dx = xvec[1]-xvec[0]
#fig0, ax0 = plt.subplots()
#rho_ss_3 = qt.steadystate(H3, c_ops3)
#rhoa3 = rho_ss_3.ptrace(0)
#W3 = qt.wigner(rhoa3, xvec, xvec)
#ax0.pcolor(xvec, xvec, W3)
#ax0.axis('equal')

psi02 = qt.tensor(psi01, qt.basis(Nb, 0))
psi03 = qt.tensor(psi01, qt.basis(Nb, 0), qt.basis(2,1))
rho03 = qt.tensor(psi01*psi01.dag(), qt.basis(Nb, 0)*qt.basis(Nb, 0).dag(),
                  qt.projection(2, 1, 1)*(1-nth)+qt.projection(2, 0, 0)*nth)

psi03 = qt.tensor(psi01, qt.basis(Nb, 0), qt.basis(2,0))

# T = 5/kappa2
T = 0.5#50/kappa2
dt = T/200.
tlist = np.linspace(0, T, T/dt+1)
if solve1mode:
    result = qt.mesolve(H1, psi01, tlist, c_ops1, [])
    at = []
    alphat = []
    p00 = []
    malphat = []
    for ii, t in enumerate(tlist):
        at.append(qt.expect(a**2, result.states[ii]))
        alphat.append(qt.expect(result.states[ii], ketalpha0))
        p00.append(qt.expect(qt.projection(Na, 0, 0), result.states[ii]))
        malphat.append(qt.expect(result.states[ii], ketmalpha0))
if solve2mode:
    fock0 = qt.basis(Na, 0)
    result2 = qt.mesolve(H2, psi02, tlist, c_ops2, [])
    pop0 = []
    at2 = []
    bt2 = []
    p00 = []
    p11 = []
    p22 = []
    alphat2 = []
    malphat2 = []
    popup = []
    popdown = []
    for ii, t in enumerate(tlist):
        at2.append(qt.expect(qt.tensor(a**2, qt.identity(Nb)),
                             result2.states[ii]))
        bt2.append(qt.expect(qt.tensor(qt.identity(Na), b**2),
                             result2.states[ii]))
        p00.append(qt.expect(qt.tensor(qt.projection(Na, 0, 0),
                                       qt.identity(Nb)),
                             result2.states[ii]))
        p11.append(qt.expect(qt.tensor(qt.projection(Na, 1, 1),
                               qt.identity(Nb)),
                     result2.states[ii]))
        p22.append(qt.expect(qt.tensor(qt.projection(Na, 2, 2),
                               qt.identity(Nb)),
                     result2.states[ii]))
        rhoa = result2.states[ii].ptrace(0)
        W2 = qt.wigner(rhoa, xvec, xvec)
        dist = np.sum(W2, axis=1)*dx

        popup.append(np.sum(dist[0:int(Nwig/2)+1])*dx)
        popdown.append(np.sum(dist[int(Nwig/2)+1:-1])*dx)
        alphat2.append(qt.expect(rhoa, ketalpha0))
        malphat2.append(qt.expect(rhoa, ketmalpha0))
        pop0.append(qt.expect(rhoa, fock0))
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

        rhoa = (projg(result3.states[ii])).ptrace(0)
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
    ax.plot(tlist, p00, label='pop 0')
    ax.plot(tlist, p11, label='pop 1')
    ax.plot(tlist, p22, label='pop 2')
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

fig3, ax3 = plt.subplots(3)
if solve1mode:
    W = qt.wigner(result.states[-1], xvec, xvec)
    ax3[0].pcolor(xvec, xvec, W)
if solve2mode:
    W2 = qt.wigner(qt.ptrace(result2.states[-1], 0), xvec, xvec)
    ax3[1].pcolor(xvec, xvec, W2)

if solve3mode:
    W3 = qt.wigner(qt.ptrace(result3.states[-1], 0), xvec, xvec)
    ax3[2].pcolor(xvec, xvec, W3)
ax3[0].axis('equal')
ax3[1].axis('equal')
ax3[2].axis('equal')

