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

    def __init__(self, wa, Za, wb, Zb, Ls, LJ, L,
                 printParams=True):
        
        # from http://arxiv.org/abs/1602.01793
        EJ, _, _ = c.convert_EJ_LJ_I0(LJ=LJ)
        
        phia_zpf = (1/phi0) * (np.sqrt((hbar/2)*Za))  # phiZPF
        na_zpf = (1/(2*e)) * (np.sqrt(hbar/2/Za))  # nZPF
        Ca = 1/(Za*wa)
        La = Za/wa
        
        phib_zpf = (1/phi0) * (np.sqrt((hbar/2)*Zb))  # phiZPF
        nb_zpf = (1/(2*e)) * (np.sqrt(hbar/2/Zb))  # nZPF
        Cb = 1/(Zb*wb)
        Lb = Zb/wb

#        omega_plasma = 2*pi*24e9
#        CJ = 1/(omega_plasma**2*LJ) # each junction has 2*LJ
#        if ECj == None:
#            ECj = e**2/2/CJ
#        self.ECj = ECj
        
        self.ELa = phi0**2/La
        self.ECa = e**2/2/Ca
        
        self.ELb = phi0**2/Lb
        self.ECb = e**2/2/Cb
        
        self.ELs = phi0**2/Ls
        self.ELc = phi0**2/L
        
        self.EJ = EJ

        self.varying_params={'pext_1':0, 'pext_2':0}
#        self.phi_ext_0 = 0
#        self.varying_params={'ECca':0}
        
        self.U_str = '2*(ELa/2/hbar*pLa**2) \
                      + ELb/2/hbar*pLb**2 \
                      + 2*(ELs/2/hbar*((pext_1+pext_2+p2-p1)/2)**2) \
                       + ELc/2/hbar*((pext_1-pext_2+p2+p1)/2)**2 \
                      - (EJ/hbar)*cos(p1) \
                       - (EJ/hbar)*cos(p2)'
                 
        self.T_str = '2*((1/16.)*(hbar/ECa)*(-dpLa-(dp2-dp1)/2)**2) \
                      + (1/16.)*(hbar/ECb)*(-dpLb-(dp2+dp1)/2)**2'


                

        if printParams:
            print("La = "+str(La*1e9)+" nH")
            print("Ca = "+str(Ca*1e15)+" fF")
            print("phia_zpf = "+str(phia_zpf))
            print("na_zpf = "+str(na_zpf))
            
            print("Lb = "+str(Lb*1e9)+" nH")
            print("Cb = "+str(Cb*1e15)+" fF")
            print("phib_zpf = "+str(phib_zpf))
            print("nb_zpf = "+str(nb_zpf))
            
            print("ELa/h = "+str(1e-9*self.ELa/hbar/2/pi)+" GHz")
            print("ECa/h = "+str(1e-9*self.ECa/hbar/2/pi)+" GHz")
            print("ELb/h = "+str(1e-9*self.ELb/hbar/2/pi)+" GHz")
            print("ECb/h = "+str(1e-9*self.ECb/hbar/2/pi)+" GHz")
            

            print("EJ/h = "+str(1e-9*self.EJ/hbar/2/pi)+" GHz")
            print("EL/h = "+str(1e-9*self.ELc/hbar/2/pi)+" GHz")
        
        self.max_order = 4
        super().__init__()

    def get_freqs_kerrs(self, particulars=None, return_components=False, max_solutions=1, sort=True, **kwargs): #particulars should be list of tuple
        res = self.get_normal_mode_frame(sort=sort, **kwargs)
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

    def get_freqs_only(self, sort=True,  **kwargs):
        res = self.get_normal_mode_frame(sort=sort, **kwargs)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi
        return fs
