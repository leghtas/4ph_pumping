# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:13:01 2017

@author: leghtas
"""
import qutip as qt
import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
import numpy.linalg as nl
from scipy.optimize import minimize, least_squares
import numdifftools as nd
from scipy.misc import derivative
import circuit as c
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

hbar=c.hbar
pi=c.pi
phi0=c.phi0
e=c.e

class CircuitSnailPA(c.Circuit):

    def __init__(self, EC, EL, EJ, alpha, n, NN=1, ECj = None,
                 printParams=True):
        
        # from http://arxiv.org/abs/1602.01793
        w, Z, LJ = c.get_w_Z_LJ_from_E(EC, EL, EJ)
        phi = (1/phi0) * (np.sqrt((hbar/2)*Z))  # phiZPF
        n_zpf = (1/(2*e)) * (np.sqrt(hbar/2/Z))  # nZPF
        C = 1/(Z*w)
        L = Z/w
        omega_plasma = 2*pi*24e9
        CJ = 1/(omega_plasma**2*(2*LJ)) # each junction has 2*LJ
        if ECj == None:
            ECj = e**2/2/CJ
            
        # Fixed parameters should be given as class attribute
        self.ECj = ECj
        self.EL = phi0**2/L
        self.EC = e**2/2/C
        self.EJ = EJ
        self.n = n
        self.NN = NN
        self.alpha = alpha
        
        # Varying parameters should be stored in this dictionary
        self.varying_params={'phi_ext_0':0}

## Symmetric
        self.U_str = 'EL/hbar*pr**2 \
                 -NN*alpha*(EJ/hbar)*cos(ps/NN) \
                 -NN*n*(EJ/hbar)*cos((NN*phi_ext_0-ps)/n/NN)'
 
### Squid if n = 2                
#        self.U_str = 'EL/hbar*pr**2 \
#                 -NN*alpha*(EJ/hbar)*cos(ps/NN) \
#                 -NN*n*(EJ/hbar)*cos((NN*phi_ext_0-ps)/n)'  

        self.T_str = '(1/32.)*(hbar/EC)*(2*dpr-dps)**2 \
                + (1/16.)*(1/NN)*(alpha+1/n)*(hbar/ECj)*(dps)**2'
                

        if printParams:
            print("w = "+str(w/2/np.pi*1e-9)+"GHz")
            print("Z = "+str(Z)+"Ohm")
            print("L = "+str(L*1e9)+" nH")
            print("C = "+str(C*1e15)+" fF")
            print("LJ = "+str(LJ*1e9)+" nH")
            print("exp_f = "+str(1/((L+LJ)*C)**0.5*1e-9)+" GHz")
            print("EL/h = "+str(1e-9*self.EL/hbar/2/pi)+" GHz")
            print("EC/h = "+str(1e-9*self.EC/hbar/2/pi)+" GHz")
            print("EJ/h = "+str(1e-9*self.EJ/hbar/2/pi)+" GHz")
            print("phi_zpf = "+str(phi))
            print("n_zpf = "+str(n_zpf))
            print("CJ per junction = "+str(CJ*1e15)+str(" fF"))
            print('')
#            print("kappab/kappaa limited by CJ = "+str(1/kappaa_over_kappab))
        
        # Maximum order of the expansion. 4 for Kerr terms
        self.max_order = 4
        super().__init__()


    def get_freqs_kerrs(self, **kwargs):

        res = self.get_normal_mode_frame(**kwargs)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi

        # calculate Kerrs from polynomial approximation of potential

        Hess2U = self.get_HessnL('U', 2,  **kwargs)
        Hess3U = self.get_HessnL('U', 3,  **kwargs)
        Hess4U = self.get_HessnL('U', 4,  **kwargs)

        Hess2_r = Hess2U([res1[0], res1[1]], P=P)
        Hess3_r = Hess3U([res1[0], res1[1]], P=P)
        Hess4_r = Hess4U([res1[0], res1[1]], P=P)

        popt2 = np.array([Hess2_r[0, 0]/2, Hess2_r[1, 1]/2])
        popt3 = np.array([Hess3_r[0, 0, 0]/6, Hess3_r[1, 1, 1]/6]) # coeff devant le phi**3
        popt4 = np.array([Hess4_r[0, 0, 0, 0]/24, Hess4_r[1, 1, 1, 1]/24]) # coeff devant le phi**4


        ZPF = popt2**(-1./4)

        Xi2 = popt2*(ZPF**2)/2/np.pi # freq en Hz
        Xi3 = 2 * popt3*(ZPF**3)/2/np.pi #coeff devant a^2.a^+
        Xi4 = 6 * popt4*(ZPF**4)/2/np.pi #coeff devant a^2.a^+2

        check_Xi2 = w2**0.5/2/np.pi

        return res1, res2, Xi2, Xi3, Xi4, check_Xi2

    def get_freqs_only(self,  **kwargs):
        res = self.get_normal_mode_frame(**kwargs)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi
        return fs
