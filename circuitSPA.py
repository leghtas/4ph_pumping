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

class CircuitSPA(c.Circuit):

    def __init__(self, EL, EJ, EC, ECJ, NN, nn, alpha,
                 printParams=True):

        self.EL = EL
        self.EJ = EJ
        self.EC = EC
        self.ECJ = ECJ
        self.NN = NN
        self.nn = nn
        self.alpha = alpha
        self.varying_params={'pext':0}
#        self.phi_ext_0 = 0
#        self.varying_params={'ECca':0}
        
        self.U_str = '0.5*(EL/hbar)*(pC-pJ)**2 \
                      - NN*alpha*(EJ/hbar)*cos(pJ/NN) \
                      - NN*nn*(EJ/hbar)*cos((NN*pext-pJ)/nn/NN)'
                 
        self.T_str = '(1/16.)*(hbar/EC)*(dpC)**2 \
                      + (1/16.)*(hbar/ECJ)*(dpJ)**2'
                      
#        self.U_str = '0.5*(EL/hbar)*(pC)**2-(EJ/hbar)*(cos(pJ-pext)+cos(pJ))'
#                 
#        self.T_str = '(1/16.)*(hbar/EC)*(dpC)**2+(1/16.)*(hbar/ECJ)*(dpJ)**2'
        
        self.max_order = 4
        super().__init__()
