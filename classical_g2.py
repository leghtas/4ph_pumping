# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:57:15 2018

@author: checkhov
"""

from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import fmin
import math
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

kappa_s = 1. / 2.
kappa_r = 2 * np.pi * 3.
g2 = 2 * np.pi * 0.1
eps_d = 50

Nd = 101
Np = 101
dets_d = np.linspace(-50,50,Nd)
dets_p = np.linspace(-50,50,Np)

#def equations(p):
#    rea_r, rea_s = p
##    print(f'rea_r{rea_r}')
##            print(ima_r)
#    a_r = rea_r #ima_r
#    a_s = rea_s #ima_s
#    return (np.abs(-1j*det_d*a_r-1j*g2*a_s**2-1j*eps_d-kappa_r/2*a_r), np.abs(-1j*(det_p+det_d)/2*a_s-2j*g2*np.conjugate(a_s)*a_r-kappa_s/2*a_s))
#
#abs_r_show = np.empty((Np, Nd))
#
#for jj, det_p in enumerate(dets_p):
#    for ii, det_d in enumerate(dets_d):
#
#
#
#        rea_r_sol, rea_s_sol =  fsolve(equations, (0, 0)) # ima_r_sol, , ima_s_sol
#        abs_r_show[jj, ii] = np.abs(rea_r_sol)#+ 1j*ima_r_sol
#        print(equations((rea_r_sol, rea_s_sol)))#, ima_s_sol,, ima_r_sol


#ax[0].pcolor(dets_d, dets_p, abs_r_show)

fig, ax = plt.subplots(3)
def complex_equations(p):
    rea_r, ima_r, rea_s, ima_s = p
#    print(f'rea_r{rea_r}')
#            print(ima_r)
    a_r = rea_r +1j*ima_r
    a_s = rea_s +1j*ima_s
    cost = np.abs(-1j*det_d*a_r-1j*g2*a_s**2-1j*eps_d-kappa_r/2*a_r) + np.abs(-1j*(det_p+det_d)/2*a_s-2*1j*g2*np.conjugate(a_s)*a_r-kappa_s/2*a_s)
#    print(cost)
    return cost

ar_show = np.empty((Np, Nd))
as_show = np.empty((Np, Nd))
solving_show = np.empty((Np, Nd))

#guess = a_s = -1j*eps_d/(kappa_s/2+1j*(det_d+det_p))
#        a_r = -1j*eps_d/(kappa_r/2+1j*det_d)
if False:
    for jj, det_p in enumerate(dets_p):
        for ii, det_d in enumerate(dets_d):

            a_s = -1j*eps_d/(kappa_s/2+1j*(det_d+det_p))*1
            a_r = -1j*eps_d/(kappa_r/2+1j*det_d)
            sol = fmin(complex_equations, (np.real(a_r), np.imag(a_r), np.real(a_s), np.imag(a_s)), disp=0, ftol = 1e-15)
            rea_r_sol, ima_r_sol, rea_s_sol, ima_s_sol =  sol

    #        issol_show[jj,ii] = mag_eq1
            ar_show[jj,ii] = np.abs(rea_r_sol+1j*ima_r_sol)**2
            as_show[jj,ii] = np.abs(rea_s_sol+1j*ima_s_sol)**2
            solving_show[jj,ii] = complex_equations(sol)

    print('ar_M_m', np.amax(ar_show), np.amin(ar_show))
    print('as_M_m', np.amax(as_show), np.amin(as_show))
    print('solving_M_m', np.amax(solving_show), np.amin(solving_show))
    ax[0].pcolor(dets_d, dets_p, ar_show)
    ax[1].pcolor(dets_d, dets_p, as_show, vmin=np.amin(ar_show), vmax=np.amax(ar_show))
    ax[2].pcolor(dets_d, dets_p, solving_show)

#fig2, ax2 = plt.subplots(3)
#ax2[0].plot(np.real(-1j*eps_d/(kappa_r/2+1j*dets_d)), np.imag(-1j*eps_d/(kappa_r/2+1j*dets_d)), '.')
#ax2[1].plot(dets_d, np.abs(-1j*eps_d/(kappa_r/2+1j*dets_d)))
#ax2[2].plot(dets_d, np.angle(-1j*eps_d/(kappa_r/2+1j*dets_d)))
plt.close('all')
fig, ax = plt.subplots(2)
def mod_as2_equation(p):
    rea_s, ima_s = p
    a_s = rea_s +1j*ima_s
    cost = np.abs(1/4/g2**2*(det_d-1j*kappa_r/2)*(det_p+det_d-1j*kappa_s)-eps_d/g2*np.exp(-2*1j*np.angle(a_s))-np.abs(a_s)**2)
    return cost

if True:
    for jj, det_p in enumerate(dets_p):
        for ii, det_d in enumerate(dets_d):

            z1 = 1/4/g2**2*(det_d-1j*kappa_r/2)*(det_p+det_d-1j*kappa_s)
            m2 = eps_d/g2
            i1 = np.imag(z1)
            sol=0
            if np.abs(i1)> m2:
                sol = None
            elif np.abs(z1)>m2:
                if np.real(z1)>0:
                    theta1 = np.arcsin(i1/m2)

                    if i1>=0:
                        theta2 = np.pi-np.arcsin(i1/m2)
                    else:
                        theta2 = -np.pi-np.arcsin(i1/m2)
#                    print(f'theta2 {theta2}')
                else:
                    sol=None
            else:
                theta1=None
                if i1>=0:
                    theta2 = np.pi-np.arcsin(i1/m2)
                else:
                    theta2 = -np.pi-np.arcsin(i1/m2)


            if sol is not None:
                if theta1 is None:
                    theta = theta2
                else:
                    theta = theta1

                theta_s = -theta/2
                theta_s = -theta/2 + np.pi

                mod2_s = z1-m2*np.exp(1j*theta)
#                print(z1-m2*np.exp(1j*theta))
                if np.imag(mod2_s)/np.real(mod2_s)>0.001:
                    raise ValueError('Found imag value')
                mod_s = np.real(mod2_s)**0.5
                a_s = np.exp(1j*theta_s)*mod_s
                a_r = (-det_p-det_d+1j*kappa_s)/4/g2*np.exp(2*1j*theta_s)

                cost = mod_as2_equation((np.real(a_s), np.imag(a_s)))




#                sol = fmin(mod_as2_equation, (np.real(a_s), np.imag(a_s)), disp=0, ftol = 1e-15)
#                rea_s_sol, ima_s_sol =  sol



            else:
                a_s=0
                a_r = -1j*eps_d/(kappa_r/2+1j*det_d)

#            sol = fmin(complex_equations, (np.real(a_r), np.imag(a_r), np.real(a_s), np.imag(a_s)), disp=0)
#
#            rea_r_sol, ima_r_sol, rea_s_sol, ima_s_sol =  sol
#            ar_show[jj,ii] = np.abs(rea_r_sol+1j*ima_r_sol)**2
            ar_show[jj,ii] = np.abs(a_r)**2
            as_show[jj,ii] = np.angle(a_r)
            solving_show[jj,ii] = cost



    print('ar_M_m', np.amax(ar_show), np.amin(ar_show))
    print('as_M_m', np.amax(as_show), np.amin(as_show))
    print('solving_M_m', np.amax(solving_show), np.amin(solving_show))
    ax[0].pcolor(dets_d, dets_p, ar_show)
    ax[1].pcolor(dets_d, dets_p, as_show)
#    ax[2].pcolor(dets_d, dets_p, solving_show)





