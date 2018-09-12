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

    def __init__(self, wa, Za, Cc, w, EJ, alpha, n, ECj = None,
                 printParams=True):
        
        # from http://arxiv.org/abs/1602.01793
        _, _, LJ = c.get_w_Z_LJ_from_E(1, 1, EJ)
        Leq = 1/(alpha+1/n)*LJ 
        _, _, Eeq = c.get_E_from_w(1, 1, Leq) 
        
        phia_zpf = (1/phi0) * (np.sqrt((hbar/2)*Za))  # phiZPF
        na_zpf = (1/(2*e)) * (np.sqrt(hbar/2/Za))  # nZPF
        Ca = 1/(Za*wa)
        La = Za/wa
        self.ELa = phi0**2/La
        self.ECa = e**2/2/Ca
        
        self.ECc = e**2/2/Cc

#        omega_plasma = 2*pi*24e9
#        CJ = 1/(omega_plasma**2*LJ) # each junction has 2*LJ
#        if ECj == None:
#            ECj = e**2/2/CJ
#        self.ECj = ECj
        
        self.EJ = EJ
        self.EJ = EJ
        self.EC = (hbar*w)**2/8/Eeq
        self.n = n
        self.alpha = alpha
        self.varying_params={'phi_ext_0':0}
#        self.phi_ext_0 = 0
#        self.varying_params={'ECca':0}
        
        self.U_str = 'ELa/2/hbar*pa**2 \
                      - alpha*(EJ/hbar)*cos(ps) \
                      - n*(EJ/hbar)*cos((phi_ext_0-ps)/n)'
#                      - EJ/hbar*cos(ps)+0*phi_ext_0'
                 
        self.T_str = '(1/16.)*(hbar/ECa)*(dpa)**2 \
                      + (1/16.)*(hbar/EC)*(dps)**2\
                      + (1/16.)*(hbar/ECc)*(dps-dpa)**2'
#                      + (1/16.)*(alpha+1/n)*hbar*(1/ECj+1/ECc)*(dpc)**2 \
#                      + (1/16.)*(hbar/ECca)*(dpca)**2 \
#                      + (1/16.)*(hbar/ECcb)*(dpc+dpca+dpa-dpb)**2'
                      # + (1/16.)*(hbar/ECca)*(dpc+dpca+dpa-dpb)**2'

                

        if printParams:
            print("fa = "+str(wa/2/pi*1e-9)+"GHz")
            print("Za = "+str(Za)+"Ohm")
            print("La = "+str(La*1e9)+" nH")
            print("Ca = "+str(Ca*1e15)+" fF")
            print("phia_zpf = "+str(phia_zpf))
            print("na_zpf = "+str(na_zpf))
            
            print("Cc = "+str(Cc*1e15)+" fF")
            
            print("ELa/h = "+str(1e-9*self.ELa/hbar/2/pi)+" GHz")
            print("ECa/h = "+str(1e-9*self.ECa/hbar/2/pi)+" GHz")
            
            print('Leq = %.3f nH'%(Leq/1e-9))
            print("LJ_grosse = "+str(LJ*1e9)+" nH")
            print("LJ_petite = "+str(LJ/alpha*1e9)+" nH")
#            print("CJ = "+str(CJ*1e15)+" fF")
#            print("fJ = "+str(1/(LJ*CJ)**0.5*1e-9/2/pi)+" GHz")
            
#            LJ_snail = 1/(alpha+1/n)*LJ
#            CJ_snail = (alpha+1/n)*CJ
#            print("LJ_snail = "+str(LJ_snail*1e9)+" nH")
#            print("CJ_snail = "+str(CJ_snail*1e15)+" fF")
#            print("f_snail = "+str(1/(LJ_snail*CJ_snail)**0.5*1e-9/2/pi)+" GHz")
            
            print("EJ/h = "+str(1e-9*self.EJ/hbar/2/pi)+" GHz")

#            print("CJ per junction = "+str(CJ*1e15)+str(" fF"))
            
            

#            print("kappab/kappaa limited by CJ = "+str(1/kappaa_over_kappab))
        
        self.max_order = 4
        super().__init__()

    def get_freqs_kerrs(self, particulars=None, return_components=False, **kwargs): #particulars should be list of tuple

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
        
        popt2 = np.array([Hess2_r[0, 0]/2, Hess2_r[1, 1]/2])
        popt3 = np.array([Hess3_r[0, 0, 0]/6, Hess3_r[1, 1, 1]/6])
        popt4 = np.array([Hess4_r[0, 0, 0, 0]/24, Hess4_r[1, 1, 1, 1]/24])
        
        ZPF = popt2**(-1./4)/4**0.5 # hbar*w/2 is the term in front of a^+.a in the hamiltonian coming from phi**2, the other half is from dphi**2 

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
            
        # factor 4 see former remark so in front of phi**2 we got w/4 (one 2 come from developping (a^+ + a)**2, the other from the kinetic part)
        Xi2 = 4 * popt2*(ZPF**2)/2/np.pi # freq en Hz : coeff devant a^+.a (*2 to get whole freq)
        Xi3 = 3 * popt3*(ZPF**3)/2/np.pi #coeff devant a^2.a^+
        Xi4 = 6 * popt4*(ZPF**4)/2/np.pi #coeff devant a^2.a^+2
#        check_Xi2 = w2**0.5/2/np.pi
        if return_components:
            return res1, res2, Xi2, Xi3, Xi4, Xip, P
        else:
            return res1, res2, Xi2, Xi3, Xi4, Xip

    def get_freqs_only(self,  **kwargs):
        res = self.get_normal_mode_frame(**kwargs)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi
        return fs
