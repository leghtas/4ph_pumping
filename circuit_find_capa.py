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
from scipy.misc import derivative, factorial
import circuit as c
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

def get_factor(which):
    factor = 1
    for elt in set(which):
        factor = factor*factorial(which.count(elt))
    return factor

hbar=c.hbar
pi=c.pi
phi0=c.phi0
e=c.e

class CircuitPump2Snail(c.Circuit):

    def __init__(self, w, Z, Cc, EJ, CJ, printParams=True):
        
        # from http://arxiv.org/abs/1602.01793
        _, _, LJ = c.get_w_Z_LJ_from_E(1, 1, EJ)
        
        C = 1/(Z*w)
        L = Z/w
        self.EL = phi0**2/L
        self.EC = e**2/2/C
        self.ECJ = e**2/2/CJ
        
        self.ECc = e**2/2/Cc
    
        self.EJ = EJ
        
        self.varying_params={'phi_ext_0':0}
        
        self.U_str = 'EL/2/hbar*pa**2 \
                      +EJ/2/hbar*pj**2 \
                      + 0*pcc*phi_ext_0'
                 
        self.T_str = '(1/16.)*(hbar/ECc)*(dpj+dpcc-dpa)**2 \
                      + (1/16.)*(hbar/ECc)*(dpcc)**2\
                      + (1/16.)*(hbar/EC)*(dpa)**2\
                      + (1/16.)*(hbar/ECJ)*(dpj)**2'

                

        if printParams:
            print("f = "+str(w/2/pi*1e-9)+"GHz")
            print("Z = "+str(Z)+"Ohm")
            print("L = "+str(L*1e9)+" nH")
            print("C = "+str(C*1e15)+" fF")
            print("LJ = "+str(LJ*1e9)+ "nH")
            
            print("Cc = "+str(Cc*1e15)+" fF")
            
            print("EL/h = "+str(1e-9*self.EL/hbar/2/pi)+" GHz")
            print("EC/h = "+str(1e-9*self.EC/hbar/2/pi)+" GHz")
            
            print('\nExpectations')
            print('fmem_nocapa = %.3f GHz\n'%(w/2/np.pi/1e9)+'fmem = %.3f GHz'%(1/(L*(C+Cc))**0.5/2/np.pi/1e9))
            print('fsna_no_capa = %.3f GHz\n'%(1/(LJ*CJ)**0.5/2/np.pi/1e9)+'fsna = %.3f GHz\n'%(1/(LJ*(Cc/2+CJ))**0.5/2/np.pi/1e9))
            
        
        self.max_order = 4
        super().__init__()

    def get_freqs_kerrs(self, particulars=None, **kwargs): #particulars should be list of tuple

        res = self.get_normal_mode_frame(**kwargs)
        res1, res2, P, w2 = res

        fs = np.sqrt(w2)/2/np.pi

        # calculate Kerrs from polynomial approximation of potential

        Hess2U = self.get_HessnL('U', 2,  **kwargs)
        Hess3U = self.get_HessnL('U', 3,  **kwargs)
        Hess4U = self.get_HessnL('U', 4,  **kwargs)

        Hess2_r = Hess2U(res1, P=P)
        Hess3_r = Hess3U(res1, P=P)
        Hess4_r = Hess4U(res1, P=P)
        
        popt2 = np.array([Hess2_r[0, 0]/2, Hess2_r[1, 1]/2, Hess2_r[2, 2]/2])
        popt3 = np.array([Hess3_r[0, 0, 0]/6, Hess3_r[1, 1, 1]/6, Hess3_r[2, 2, 2]/6])
        popt4 = np.array([Hess4_r[0, 0, 0, 0]/24, Hess4_r[1, 1, 1, 1]/24, Hess4_r[2, 2, 2, 2]/24])
        
        ZPF = popt2**(-1./4)
        

        if particulars is not None:
            Xip = []
            for particular in particulars:
                factor = get_factor(particular)
                if len(particular)==2:
                    Hessp_r = Hess2_r
                    poptp = Hessp_r[particular[0], particular[1]]/factor
                    Xip.append(poptp*(ZPF[particular[0]]*ZPF[particular[0]])/2/np.pi)
                elif len(particular)==3:
                    Hessp_r = Hess3_r       
                    poptp = Hessp_r[particular[0], particular[1], particular[2]]/factor
                    Xip.append(poptp*(ZPF[particular[0]]*ZPF[particular[1]]*ZPF[particular[2]])/2/np.pi)
                elif len(particular)==4:
                    Hessp_r = Hess4_r 
                    poptp = Hessp_r[particular[0], particular[1], particular[2], particular[3]]/factor
                    Xip.append(poptp*(ZPF[particular[0]]*ZPF[particular[1]]*ZPF[particular[2]]*ZPF[particular[3]])/2/np.pi)
        else:
            Xip = None
            
        Xi2 = popt2*(ZPF**2)/2/np.pi # freq en Hz
        Xi3 = 2 * popt3*(ZPF**3)/2/np.pi #coeff devant a^2.a^+
        Xi4 = 6 * popt4*(ZPF**4)/2/np.pi #coeff devant a^2.a^+2
#        check_Xi2 = w2**0.5/2/np.pi

        return res1, res2, Xi2, Xi3, Xi4, Xip

    def get_freqs_only(self,  **kwargs):
        res = self.get_normal_mode_frame(**kwargs)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi
        return fs
