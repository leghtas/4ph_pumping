# -*- coding: utf-8 -*-
"""
Created on Tue Jan  23 17:16:36 2018

@author: leghtas
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import os
import time

plt.close('all')
plt.ion()

solve1mode = True
solve2mode = True
solve3mode = True

params = [{'kappaao2pi' : 0.053, 'kappabo2pi' : 10, 'kappaphio2pi' : 0.05, 'Kerro2pi' : 0.05, 'g2o2pi' : 0.5 },
          {'kappaao2pi' : 0.053, 'kappabo2pi' : 10, 'kappaphio2pi' : 0.05, 'Kerro2pi' : 0.05, 'g2o2pi' : 1 },
          {'kappaao2pi' : 0.053, 'kappabo2pi' : 10, 'kappaphio2pi' : 0.05, 'Kerro2pi' : 0.05, 'g2o2pi' : 2 },
          {'kappaao2pi' : 0.053, 'kappabo2pi' : 10, 'kappaphio2pi' : 0.05, 'Kerro2pi' : 0.05, 'g2o2pi' : 3 }
          ]

for param in params:
    kappaa, kappab, kappaphi, Kerr, g2 = 2*np.pi*np.array([param['kappaao2pi'],
                                                           param['kappabo2pi'],
                                                           param['kappaphio2pi'],
                                                           param['Kerro2pi'],
                                                           param['g2o2pi']])

    chiab = 2 * np.pi *0.1*0

    # 1 mode
    kappa2 = 4*g2**2/kappab  # MHz

    # 3 mode
    chi = 2 * np.pi * 1.
    nth = 0.2
    T1 = 10

    Na = 30
    Nb = 3
    a = qt.destroy(Na)
    b = qt.destroy(Nb)

    nvec = np.arange(1, 10, 2)
    T = 10
    dt = T/500.
    tlist = np.linspace(0, T, T/dt+1)
    alpha0vec = [np.sqrt(nn)*np.exp(1j*np.pi/2) for nn in nvec]

    popups1 = []
    popups2 = []
    popups3 = []
    for ii, alpha0 in enumerate(alpha0vec):
        Na = 10 + int(nvec[ii] + 5*np.sqrt(nvec[ii]))
        a = qt.destroy(Na)
        print(Na)
        H1 = -Kerr * a.dag()**2*a**2
        H2 = (qt.tensor(H1, qt.identity(Nb)) +
              g2*(qt.tensor((a**2-alpha0**2), b.dag()) +
                  qt.tensor((a.dag()**2-alpha0.conjugate()**2), b)) -
              chiab*(qt.tensor(a.dag()*a, b.dag()*b)))
        H3 = qt.tensor(H2, qt.identity(2)) + chi * qt.tensor(a.dag()*a, qt.identity(Nb),
                       (qt.sigmaz()+qt.identity(2))/2)

        ketalpha0 = qt.coherent(Na, alpha0)
        ketmalpha0 = qt.coherent(Na, -alpha0)

        C_alpha0 = qt.coherent(Na, alpha0) + qt.coherent(Na, -alpha0)
        C_alpha0 = C_alpha0 / C_alpha0.norm()

        C_malpha0 = qt.coherent(Na, alpha0) - qt.coherent(Na, -alpha0)
        C_malpha0 = C_malpha0 / C_malpha0.norm()

        psi01 = (C_alpha0 - C_malpha0) / np.sqrt(2)
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

        #T = 5/kappa2

        if solve1mode:
            result = qt.mesolve(H1, psi01, tlist, c_ops1, [])
            at = []
            alphat = []
            malphat = []
            popup = []
            popdown = []
            for ii, t in enumerate(tlist):
                at.append(qt.expect(a**2, result.states[ii]))
                alphat.append(qt.expect(result.states[ii], ketalpha0))
                malphat.append(qt.expect(result.states[ii], ketmalpha0))
                rhoa = result.states[ii]
                W1 = qt.wigner(rhoa, xvec, xvec)
                dist = np.sum(W1, axis=1)*dx

                popup.append(np.sum(dist[0:int(Nwig/2)+1])*dx)
                popdown.append(np.sum(dist[int(Nwig/2)+1:-1])*dx)
            popups1.append(popup)

        if solve2mode:
            fock0 = qt.basis(Na, 0)
            result2 = qt.mesolve(H2, psi02, tlist, c_ops2, [])
            pop0 = []
            at2 = []
            bt2 = []
            alphat2 = []
            malphat2 = []
            popup = []
            popdown = []
            for ii, t in enumerate(tlist):
                at2.append(qt.expect(qt.tensor(a**2, qt.identity(Nb)),
                                     result2.states[ii]))
                bt2.append(qt.expect(qt.tensor(qt.identity(Na), b**2),
                                     result2.states[ii]))
                rhoa = result2.states[ii].ptrace(0)
                W2 = qt.wigner(rhoa, xvec, xvec)
                dist = np.sum(W2, axis=1)*dx

                popup.append(np.sum(dist[0:int(Nwig/2)+1])*dx)
                popdown.append(np.sum(dist[int(Nwig/2)+1:-1])*dx)
                alphat2.append(qt.expect(rhoa, ketalpha0))
                malphat2.append(qt.expect(rhoa, ketmalpha0))
                pop0.append(qt.expect(rhoa, fock0))
            popups2.append(popup)
        if solve3mode:
            result3 = qt.mesolve(H3, rho03, tlist, c_ops3, [])
            at3 = []
            bt3 = []
            alphat3 = []
            malphat3 = []
            popup = []
            popdown = []
            for ii, t in enumerate(tlist):
                at3.append(qt.expect(qt.tensor(a**2, qt.identity(Nb), qt.identity(2)),
                                     result3.states[ii]))
                bt3.append(qt.expect(qt.tensor(qt.identity(Na), b**2, qt.identity(2)),
                                     result3.states[ii]))
                rhoa = result3.states[ii].ptrace(0)
                W3 = qt.wigner(rhoa, xvec, xvec)
                dist = np.sum(W3, axis=1)*dx
                popup.append(np.sum(dist[0:int(Nwig/2)+1])*dx)
                popdown.append(np.sum(dist[int(Nwig/2)+1:-1])*dx)
                alphat3.append(qt.expect(rhoa, ketalpha0))
                malphat3.append(qt.expect(rhoa, ketmalpha0))
            popups3.append(popup)


    if False:
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

    fig, ax = plt.subplots(3)
    for ii in range(len(popups1)):
        ax[0].plot(tlist, popups1[ii], '.-')
        ax[1].plot(tlist, popups2[ii], '.-')
        if solve3mode:
            ax[2].plot(tlist, popups3[ii])

    popups1_T = np.zeros(len(popups1))
    for ii in range(len(popups1_T)):
        popups1_T[ii]=popups1[ii][-1]

    popups2_T = np.zeros(len(popups2))
    for ii in range(len(popups2_T)):
        popups2_T[ii]=popups2[ii][-1]

    if solve3mode:
        popups3_T = np.zeros(len(popups3))
        for ii in range(len(popups3_T)):
            popups3_T[ii]=popups3[ii][-1]


    kappaphi_n = [kappaphi*n/(2*np.sinh(2*n)) for n in nvec]

    fig1, ax1 = plt.subplots(1)
    ax1.plot(nvec,-np.log10(-np.log(2*popups2_T-1)/tlist[-1]), '.-',  label='2 mode')
    ax1.plot(nvec,-np.log10(-np.log(2*popups1_T-1)/tlist[-1]), '.-', label='1 mode')
    ax1.plot(nvec,-np.log10(kappaphi_n), '.-', label='analytics')
    ax1.legend()
    tstamp = round(time.time())
    title = 'g2/2pi = %s, Kerr/2pi = %s, kappaa/2pi = %s, \n kappab/2pi = %s, kappaphi/2pi = %s, tstamp = %s ' % \
            (np.round((g2/2/np.pi), 2), np.round((Kerr/2/np.pi), 2), np.round((kappaa/2/np.pi), 2),
             np.round((kappab/2/np.pi), 2), np.round((kappaphi/2/np.pi), 2), tstamp)
    ax1.set_title(title)
    ax1.set_xlabel('nbar')
    ax1.set_ylabel('log10(1/gamma[us])')
    tosave = ((((title.replace('/','o')).replace(' ','')).replace(',','_')).replace('.','p')).replace('\n','')
    ax1.set_ylim([0,9])
    ax1.set_xlim([nvec[0]-1,nvec[-1]+1])
    plt.savefig('images/'+tosave+'.png')

    savedir = str(tstamp)
    os.mkdir(savedir)
    savestr = f'g2o2pi_{g2/2/np.pi}chiabo2pi{chiab/2/np.pi}'
    if solve3mode:
        np.save(savedir+'/popups3.npy', popups3)
    np.save(savedir+'/popups2.npy', popups2)
    np.save(savedir+'/popups1.npy', popups1)
    np.save(savedir+'/tlist.npy', tlist)
    np.save(savedir+'/nvec.npy', np.arange(0.5, 10, 0.5))

    #tlist = np.load('tlist.npy')

    #n_old = np.load('nvec.npy')
    #pop2_old = np.load('popups2.npy')
    #popups2_T_old = np.zeros(len(pop2_old))
    #for ii in range(len(popups2_T_old)):
    #    popups2_T_old[ii]=pop2_old[ii][-1]
    #ax1.plot(n_old,np.log(-np.log(1-2*popups2_T_old)))