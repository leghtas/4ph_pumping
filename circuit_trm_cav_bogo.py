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

class CircuitTrmCav(c.Circuit):

    def __init__(self, w, EJ, wa, Za, Cc, printParams=True):
        
        # from http://arxiv.org/abs/1602.01793
        Ca = 1/(Za*wa)
        La = Za/wa
        self.ELa = phi0**2/La
        self.ECa = e**2/2/Ca
        self.ECc = e**2/2/Cc
        
        _, LJ, _ = c.convert_EJ_LJ_I0(EJ)
        self.EJ = EJ
        self.EC = (hbar*w)**2/8/EJ
        CJ = e**2/2/self.EC
        print(self.ECc)
        print(self.ECa)
        self.varying_params={}

        
        self.U_str = 'ELa/2/hbar*pa**2 \
                      + EJ/2/hbar*p**2 \
                      - EJ/24/hbar*p**4 \
                      + EJ/24/5/6/hbar*p**6'
                 
        self.T_str = '(1/16.)*(hbar/EC)*(dp)**2 \
                      + (1/16.)*(hbar/ECc)*(dp-dpa)**2 \
                      + (1/16.)*(hbar/ECa)*(dpa)**2'
            
        print('\nCoeffs')
        print(1/8*hbar/self.EC+1/8*hbar/self.ECc)
        print(1/8*hbar/self.ECc)
        print(1/8*hbar/self.ECa+1/8*hbar/self.ECc)
        
        print('\n')
        print(self.ELa/hbar)
        print(self.EJ/hbar)
        
        self.wa_t = 1/np.sqrt(La*(Ca+Cc))
        self.w_t = 1/np.sqrt(LJ*(CJ+Cc))
        self.phiZPFa_t = np.sqrt(La*hbar*self.wa_t/2)/phi0
        self.phiZPFj_t = np.sqrt(LJ*hbar*self.w_t/2)/phi0


        self.g = np.sqrt(wa*w/(4*(1+CJ/Cc)*(1+Ca/Cc)))
        
        self.phiZPFa = np.sqrt(La*hbar*wa/2)/phi0
        self.phiZPFj = np.sqrt(LJ*hbar*w/2)/phi0
        print('ELa, EJ')
        print(self.ELa, self.EJ)
        
        if printParams:
            
            print("EJ/h = "+str(1e-9*self.EJ/hbar/2/pi)+" GHz")
            print("EC/h = "+str(1e-9*self.EC/hbar/2/pi)+" GHz")
            print(1/np.sqrt(1/8*hbar/self.EC))
            print(1/(1/8*hbar/self.ECa)*self.ELa/hbar)
            print(1/(1/8*hbar/self.EC)*self.EJ/hbar)
        
        self.max_order = 4
        super().__init__()