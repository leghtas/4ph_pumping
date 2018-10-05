#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:57:03 2018

@author: Vil
"""

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

class CircuitTunableReso(c.Circuit):

    def __init__(self, wa, Za, EJ, alpha, n, ES, ECj = None,
                 printParams=True):
        
        # from http://arxiv.org/abs/1602.01793
                
        phia_zpf = (1/phi0) * (np.sqrt((hbar/2)*Za))  # phiZPF
        na_zpf = (1/(2*e)) * (np.sqrt(hbar/2/Za))  # nZPF
        Ca = 1/(Za*wa)
        La = Za/wa
        self.ELa = phi0**2/La
        self.ECa = e**2/2/Ca
        
        _, _, LJ = c.get_w_Z_LJ_from_E(1, 1, EJ)
        _, _, LS = c.get_w_Z_LJ_from_E(1, 1, ES)
        omega_plasma = 2*pi*24e9
        CJ = 1/(omega_plasma**2*LJ) # each junction has 2*LJ
        CS = 1/(omega_plasma**2*LS)
        self.EJ = EJ
        self.ES = ES
        self.ECj = e**2/2/CJ
        self.ECs = e**2/2/CS/(1+alpha) # accounting for the two parasitic capa
        
        Leq = La + n*LJ # + LS/(1+alpha)
        Ceq = 1/(1/Ca) # + n/CJ) # + 1/CS/(1+alpha) 
        weq = 1/np.sqrt(Leq*Ceq)
        #_, _, Eeq = c.get_E_from_w(1, 1, Leq) 
        
        self.alpha = alpha
        self.n = n
        self.varying_params={'phi_ext_0':0}
#        self.phi_ext_0 = 0
#        self.varying_params={'ECca':0}
        
        self.U_str = 'ELa/2/hbar*pa**2'
        for i in range(self.n):
            self.U_str += '- (EJ/hbar)*cos(p'+str(i)+')'
                 
        self.T_str = '(1/16.)*(hbar/ECa)*(dpa'
        for i in range(self.n):
            self.T_str += '+ dp'+str(i)
        self.T_str += ')**2'
        for i in range(self.n):
            self.T_str += '+(1/16)*(hbar/ECj)*(dp'+str(i)+')**2'

                

        if printParams:
            print("fa = "+str(wa/2/pi*1e-9)+"GHz")
            print("Za = "+str(Za)+"Ohm")
            print("La = "+str(La*1e9)+" nH")
            print("Ca = "+str(Ca*1e15)+" fF")         
            print("ELa/h = "+str(1e-9*self.ELa/hbar/2/pi)+" GHz")
            print("ECa/h = "+str(1e-9*self.ECa/hbar/2/pi)+" GHz")      
            print("phia_zpf = "+str(phia_zpf))
            print("na_zpf = "+str(na_zpf))
            print("-------------")
            print("Leq = "+str(Leq*1e9)+" nH")
            print("Ceq = "+str(Ceq*1e15)+" fF")
            print("feq = "+str(weq/2/pi*1e-9)+"GHz")
            print("-------------")
            print("LJ = "+str(LJ*1e9)+" nH")
            print("CJ = "+str(CJ*1e15)+" fF")
            print("LS_grosse = "+str(LS*1e9)+" nH")
            print("CS_grosse = "+str(CS*1e15)+" fF")
            print("LS_petite = "+str(LS/alpha*1e9)+" nH")
            print("CS_petite = "+str(alpha*CS*1e15)+" fF")
#            print("CJ = "+str(CJ*1e15)+" fF")
#            print("fJ = "+str(1/(LJ*CJ)**0.5*1e-9/2/pi)+" GHz")
            
#            LJ_snail = 1/(alpha+1/n)*LJ
#            CJ_snail = (alpha+1/n)*CJ
#            print("LJ_snail = "+str(LJ_snail*1e9)+" nH")
#            print("CJ_snail = "+str(CJ_snail*1e15)+" fF")
#            print("f_snail = "+str(1/(LJ_snail*CJ_snail)**0.5*1e-9/2/pi)+" GHz")
            
            print("EJ/h = "+str(1e-9*self.EJ/hbar/2/pi)+" GHz")

#            print("CJ per junction = "+str(CJ*1e15)+str(" fF"))
            
            

#            print("ksappab/kappaa limited by CJ = "+str(1/kappaa_over_kappab))
        
        self.max_order = 4
        super().__init__()

    def get_freqs_kerrs(self, particulars=None, return_components=False, max_solutions=1, **kwargs): #particulars should be list of tuple
        res = self.get_normal_mode_frame(**kwargs)
        res1s, res2, Ps, w2s = res
        
        res1s = list(res1s)
        Ps = list(Ps)
        
        fs = np.sqrt(w2s)/2/np.pi

        # calculate Kerrs from polynomial approximation of potential
        Hess2U = self.get_HessnL('U', 2,  **kwargs)
        Hess3U = self.get_HessnL('U', 3,  **kwargs)
        Hess4U = self.get_HessnL('U', 4,  **kwargs)
        
        Xi2s = []
        Xi3s = []
        Xi4s = []
        Xips = []
        for res1, P in zip(res1s, Ps):
            Hess2_r = Hess2U(res1, P=P)
            Hess3_r = Hess3U(res1, P=P)
            Hess4_r = Hess4U(res1, P=P)
            
            popt2 = np.array([Hess2_r[ii, ii]/2 for ii in range(self.dim)])
            popt3 = np.array([Hess3_r[ii, ii, ii]/6 for ii in range(self.dim)])
            popt4 = np.array([Hess4_r[ii, ii, ii, ii]/24 for ii in range(self.dim)])
            
            ZPF = popt2**(-1./4)/4**0.5 # hbar*w/2 is the term in front of a^+.a in the hamiltonian coming from phi**2, the other half is from dphi**2 
    
            if particulars is not None:
                Xip = []
                for particular in particulars:
                    factor = get_factor(particular)
                    if len(particular)==2:
                        poptp = Hess2_r[particular]/factor
                        Xip.append(poptp*(ZPF[particular[0]]*ZPF[particular[0]])/2/np.pi)
                    elif len(particular)==3:
                        poptp = Hess3_r[particular]/factor
                        Xip.append(poptp*(ZPF[particular[0]]*ZPF[particular[1]]*ZPF[particular[2]])/2/np.pi)
                    elif len(particular)==4:
                        poptp = Hess4_r[particular]/factor
                        Xip.append(poptp*(ZPF[particular[0]]*ZPF[particular[1]]*ZPF[particular[2]]*ZPF[particular[3]])/2/np.pi)
                Xip = np.array(Xip)
            else:
                Xip = None
                
            # factor 4 see former remark so in front of phi**2 we got w/4 (one 2 come from developping (a^+ + a)**2, the other from the kinetic part)
            Xi2 = 4 * popt2*(ZPF**2)/2/np.pi # freq en Hz : coeff devant a^+.a (*2 to get whole freq)
            Xi3 = 3 * popt3*(ZPF**3)/2/np.pi #coeff devant a^2.a^+
            Xi4 = 6 * popt4*(ZPF**4)/2/np.pi #coeff devant a^2.a^+2
            
            Xi2s.append(Xi2)
            Xi3s.append(Xi3)
            Xi4s.append(Xi4)
            Xips.append(Xip)
    #        check_Xi2 = w2**0.5/2/np.pi
        n_solutions = len(Xi2s)
        if len(Xi2s)<max_solutions:
            for ii, item in enumerate([Xi2s, Xi3s, Xi4s, res1s, Xips, Ps]):
                if item[0] is not None:
                    to_add = np.nan*np.ones(item[0].shape)
                    to_adds = [to_add]*(max_solutions-n_solutions)
                    item += to_adds
                else:
                    to_adds = [None]*(max_solutions-n_solutions)
                    item += to_adds
                    
        return_list = [Xi2s, Xi3s, Xi4s, res1s, Xips, Ps]
        for ii, item in enumerate(return_list):
            return_list[ii] = np.array(item[:max_solutions])
        Xi2s, Xi3s, Xi4s, res1s, Xips, Ps = return_list
                
        if return_components:
            return res1s, res2, Xi2s, Xi3s, Xi4s, Xips, Ps
        else:
            return res1s, res2, Xi2s, Xi3s, Xi4s, Xips

    def get_freqs_only(self,  **kwargs):
        res = self.get_normal_mode_frame(**kwargs)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi
        return fs
