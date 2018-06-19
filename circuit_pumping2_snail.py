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

    def __init__(self, wa, Za, wb, Zb, Cca, Ccb, Cc, EJ, alpha, n, ECj = None,
                 printParams=True):
        
        # from http://arxiv.org/abs/1602.01793
        _, _, LJ = c.get_w_Z_LJ_from_E(1, 1, EJ)
        
        phia_zpf = (1/phi0) * (np.sqrt((hbar/2)*Za))  # phiZPF
        na_zpf = (1/(2*e)) * (np.sqrt(hbar/2/Za))  # nZPF
        Ca = 1/(Za*wa)
        La = Za/wa
        self.ELa = phi0**2/La
        self.ECa = e**2/2/Ca
        self.ECca = e**2/2/Cca
        
        
        phib_zpf = (1/phi0) * (np.sqrt((hbar/2)*Zb))  # phiZPF
        nb_zpf = (1/(2*e)) * (np.sqrt(hbar/2/Zb))  # nZPF
        Cb = 1/(Zb*wb)
        Lb = Zb/wb
        self.ELb = phi0**2/Lb
        self.ECb = e**2/2/Cb
        self.ECcb = e**2/2/Ccb
        
        self.ECc = e**2/2/Cc

        omega_plasma = 2*pi*24e9
        CJ = 1/(omega_plasma**2*LJ) # each junction has 2*LJ
        if ECj == None:
            ECj = e**2/2/CJ
        self.ECj = ECj
        
        self.EJ = EJ
        self.n = n
        self.alpha = alpha
        
        self.varying_params={'phi_ext_0':0}
        
        self.U_str = 'ELa/2/hbar*pa**2 \
                      + ELb/2/hbar*pb**2 \
                      -alpha*(EJ/hbar)*cos(pc) \
                      -n*(EJ/hbar)*cos((phi_ext_0-pc)/n) \
                      + 0.0*ELb/2/hbar*pca**2'
                 
        self.T_str = '(1/16.)*(hbar/ECa)*(dpa)**2 \
                      + (1/16.)*(hbar/ECb)*(dpb)**2\
                      + (1/16.)*(alpha+1/n)*hbar*(1/ECj+1/ECc)*(dpc)**2 \
                      + (1/16.)*(hbar/ECca)*(dpca)**2 \
                      + (1/16.)*(hbar/ECcb)*(dpc+dpca+dpa-dpb)**2'

                

        if printParams:
            print("fa = "+str(wa/2/pi*1e-9)+"GHz")
            print("Za = "+str(Za)+"Ohm")
            print("La = "+str(La*1e9)+" nH")
            print("Ca = "+str(Ca*1e15)+" fF")
            print("Cca = "+str(Cca*1e15)+" fF")
            print("phia_zpf = "+str(phia_zpf))
            print("na_zpf = "+str(na_zpf))
            
            print("fb = "+str(wb/2/pi*1e-9)+"GHz")
            print("Zb = "+str(Zb)+"Ohm")
            print("Lb = "+str(Lb*1e9)+" nH")
            print("Cb = "+str(Cb*1e15)+" fF")
            print("Ccb = "+str(Cca*1e15)+" fF")
            print("phib_zpf = "+str(phib_zpf))
            print("nb_zpf = "+str(nb_zpf))
            
            print("Cc = "+str(Cc*1e15)+" fF")
            
            print("ELa/h = "+str(1e-9*self.ELa/hbar/2/pi)+" GHz")
            print("ECa/h = "+str(1e-9*self.ECa/hbar/2/pi)+" GHz")
            print("ELb/h = "+str(1e-9*self.ELb/hbar/2/pi)+" GHz")
            print("ECb/h = "+str(1e-9*self.ECb/hbar/2/pi)+" GHz")
            
            print("LJ_grosse = "+str(LJ*1e9)+" nH")
            print("LJ_petite = "+str(LJ/alpha*1e9)+" nH")
            print("CJ = "+str(CJ*1e15)+" fF")
            print("fJ = "+str(1/(LJ*CJ)**0.5*1e-9/2/pi)+" GHz")
            
            LJ_snail = 1/(alpha+1/n)*LJ
            CJ_snail = (alpha+1/n)*CJ
            print("LJ_snail = "+str(LJ_snail*1e9)+" nH")
            print("CJ_snail = "+str(CJ_snail*1e15)+" fF")
            print("f_snail = "+str(1/(LJ_snail*CJ_snail)**0.5*1e-9/2/pi)+" GHz")
            
            print("EJ/h = "+str(1e-9*self.EJ/hbar/2/pi)+" GHz")

            print("CJ per junction = "+str(CJ*1e15)+str(" fF"))
            
            

#            print("kappab/kappaa limited by CJ = "+str(1/kappaa_over_kappab))
        
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
        
        popt2 = np.array([Hess2_r[0, 0]/2, Hess2_r[1, 1]/2, Hess2_r[2, 2]/2, Hess2_r[3, 3]/2])
        popt3 = np.array([Hess3_r[0, 0, 0]/6, Hess3_r[1, 1, 1]/6, Hess3_r[2, 2, 2]/6, Hess3_r[3, 3, 3]/6])
        popt4 = np.array([Hess4_r[0, 0, 0, 0]/24, Hess4_r[1, 1, 1, 1]/24, Hess4_r[2, 2, 2, 2]/24, Hess4_r[3, 3, 3, 3]/4])
        
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
