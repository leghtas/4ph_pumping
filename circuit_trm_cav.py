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

    def __init__(self, w, EJ, wa, Za, Cc, printParams=True):
        
        # from http://arxiv.org/abs/1602.01793
        Ca = 1/(Za*wa)
        La = Za/wa
        self.ELa = phi0**2/La
        self.ECa = e**2/2/Ca
        self.ECc = e**2/2/Cc

        self.EJ = EJ
        self.EC = (hbar*w)**2/8/EJ
        print(self.ECc)
        print(self.ECa)
        self.varying_params={'phi_ext_0':0}

        
        self.U_str = 'ELa/2/hbar*pa**2 \
                      - EJ/hbar*cos(p)+0*phi_ext_0'
                 
        self.T_str = '(1/16.)*(hbar/EC)*(dp)**2 \
                      + (1/16.)*(hbar/ECc)*(dp-dpa)**2 \
                      + (1/16.)*(hbar/ECa)*(dpa)**2'

        if printParams:
            
            print("EJ/h = "+str(1e-9*self.EJ/hbar/2/pi)+" GHz")
            print("EC/h = "+str(1e-9*self.EC/hbar/2/pi)+" GHz")
        
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
