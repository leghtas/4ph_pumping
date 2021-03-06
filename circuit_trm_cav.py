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
        
        if printParams:
            print("EJ/h = "+str(1e-9*self.EJ/hbar/2/pi)+" GHz")
            print("EC/h = "+str(1e-9*self.EC/hbar/2/pi)+" GHz")
        
        self.max_order = 4
        super().__init__()