# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:13:01 2017

@author: leghtas
"""
import qutip as qt
import scipy.constants as sc
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.linalg as sl
import numpy.linalg as nl
from scipy.optimize import minimize, least_squares
import numdifftools as nd
from scipy.misc import derivative
from circuit import *
import warnings
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

def get_symbol_list(sympy_expr):
    symbols = []
    def get_all_symbols(_sympy_expr, _symbols):
        for arg in _sympy_expr.args:
            if arg.is_Symbol:
                _symbols.append(str(arg))
            elif len(arg.args)>0:
                get_all_symbols(arg, _symbols)
    get_all_symbols(sympy_expr, symbols)
    symbols=list(set(symbols))
    return symbols

def format_diff(which, variables):
    which_format=[]
    for ii, var in enumerate(which):
        for jj in range(var):
            which_format.append(variables[ii])
    return which_format

def which_list(n, dim):
    if n==1:
        loc_list = []
        for ii in range(dim):
            loc_list.append([0 if jj!=ii else 1 for jj in range(dim)])
        return loc_list
    else:
        loc_list = which_list(1, dim)
        new_which_list = []
        for which in which_list(n-1, dim):
            for loc in loc_list:
                new_which_list.append(list(np.array(which)+np.array(loc)))
        return new_which_list
    
def tuple_list(n):
    _tuple_list = [tuple(range(n))]
    for ii in range(n-1):
        temp_list = list(range(n))
        temp_list[n-ii-2]=n-1
        temp_list[n-1]=n-ii-2
        _tuple_list.append(tuple(temp_list))
    return _tuple_list

class CircuitSnailPA(Circuit):

    def __init__(self, EC, EL, EJ, alpha, n,
                 printParams=True):
        
        self.hbar = hbar
        self.varying_params={'phi_ext_0':0}

        # from http://arxiv.org/abs/1602.01793
        w, Z, LJ = get_w_Z_LJ_from_E(EC, EL, EJ)
        phi = (1/phi0) * (np.sqrt((hbar/2)*Z))  # phiZPF
        n_zpf = (1/(2*e)) * (np.sqrt(hbar/2/Z))  # nZPF
        C = 1/(Z*w)
        L = Z/w
        self.EL = phi0**2/L
        self.EC = e**2/2/C
        self.EJ = EJ
        self.n = n
        self.alpha = alpha

        self.kept_dim = []

#        epsilonbar = pi/2/3
#        g4 = (self.EJ/2)*epsilonbar*(1./math.factorial(4))*phia**4*phib
#        g2 = (self.EJ/2)*epsilonbar*(1./math.factorial(2))*phia**2*phib

#        hbarXiaa = 0.5*(self.EJ/100.)*phia**4  # 1 % error on EJ
#        hbarXiab = (self.EJ/100.)*phia**2*phib**2

        omega_plasma = 2*pi*24e9
        CJ = 1/(omega_plasma**2*(2*LJ))  # each junction has 2*LJ
#        beta = 2*CJ/np.sqrt((Ca+CJ)*(Cb+CJ))  # 2.104 in Steve's notes
#        g = beta*np.sqrt(wa*wb)  # energy/hbar 2.110 in Steve's notes
#        kappaa_over_kappab = g**2/(wa-wb)**2  # 6.40
        #  Phia = phia*qt.tensor(a+a.dag(), qt.qeye(nb))
        #  Phib = phib*qt.tensor(qt.qeye(na), b+b.dag())

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
#            print("g4/h = "+str(1e-6*g4/hbar/2/pi)+" MHz")
#            print("g2/h = "+str(1e-6*g2/hbar/2/pi)+" MHz")
#            print("Xiaa/2pi = "+str(1e-6*hbarXiaa/hbar/2/pi)+" MHz")
#            print("Xiab/2pi = "+str(1e-6*hbarXiab/hbar/2/pi)+" MHz")
            print("CJ per junction = "+str(CJ*1e15)+str(" fF"))
#            print("kappab/kappaa limited by CJ = "+str(1/kappaa_over_kappab))
        self.prepare_U_formal()
            
    def remove_params(self, symbol_list):
        variables = []
        for symbol in symbol_list:
            if (not symbol in self.__dict__.keys()) and (not symbol in self.varying_params.keys()):
                variables.append(symbol)
        variables = sorted(variables)
        return variables
    
    def prepare_U_formal(self): 
        U_str = 'EL/hbar*pr**2 \
                 -alpha*(EJ/hbar)*cos(ps) +\
                 -n*(EJ/hbar)*cos((phi_ext_0-ps)/n)'
        U_expr = parse_expr(U_str)
#        print(U_expr)
        U_expr_symbols = get_symbol_list(U_expr)
        self.U_variables = self.remove_params(U_expr_symbols)
        self.dim = len(self.U_variables)

        U_expr_sub = U_expr.subs(self.__dict__)
        self.U_formal = U_expr_sub
#        U = sp.lambdify(('pa', 'pb', 'pc'), U_expr_sub, 'numpy')

    
    def get_anyU(self, which, parameters=None): # which should be a tuple with derivativative wanted (1,0,0) for the first one
        if parameters is None:
            parameters=self.varying_params      # parameters should be a dictionnary of variable parameters
        
        U_expr_sub = self.U_formal.subs(parameters)
        if sum(which)==0:
            _anyU = U_expr_sub
        else:
            which_format = format_diff(which, self.U_variables)
            _anyU = sp.diff(U_expr_sub, *which_format)
        _anyU = sp.lambdify(tuple(self.U_variables), _anyU, 'numpy')
        def anyU(p, P=np.identity(self.dim)):
            p=P.dot(p)
            return _anyU(*p)
        return anyU
    
    def get_HessnU(self, n, parameters=None):
        if parameters is None:
            parameters=self.varying_params
        def HessnU(p, P=np.identity(self.dim)):
#            p = P.dot(p)   
            _HessnU= []
            for which in which_list(n, self.dim):
                _HessnU.append(self.get_anyU(which, parameters=parameters)(p))
            shape=[self.dim for ii in range(n)]
            _HessnU=np.array(_HessnU).reshape(shape)
            
            permutations = tuple_list(n)
            for permutation in permutations:
                _HessnU = np.transpose(np.dot(np.transpose(_HessnU, permutation), P), permutation)
#            _HessU1 = np.transpose(np.dot(np.transpose(_HessU,(0,1)), P),(0,1))
#            _HessU2 = np.transpose(np.dot(np.transpose(_HessU1,(1,0)), P),(1,0))
#            _HessU_basis = np.dot(np.dot(P.T, _HessU), P)
#            _HessU_basis = _HessU2
#            _Hess4U1 = np.transpose(np.dot(np.transpose(_Hess4U,(0,1,2,3)), P),(0,1,2,3))
#            _Hess4U2 = np.transpose(np.dot(np.transpose(_Hess4U1,(1,0,2,3)), P),(1,0,2,3))
#            _Hess4U3 = np.transpose(np.dot(np.transpose(_Hess4U2,(2,1,0,3)), P),(2,1,0,3))
#            _Hess4U4 = np.transpose(np.dot(np.transpose(_Hess4U3,(3,1,2,0)), P),(3,1,2,0))
            return _HessnU
        return HessnU

    def get_T(self, phi_ext_0=0):
        def T(dp, P=np.identity(2)): # dp: dphi/dt
            dp = P.dot(dp)
            (dps, dpr) = (dp[0],dp[1])
            _T = (1/32.)*(hbar/self.EC)*(2*dpr-dps)**2+(1/32.)*(hbar/self.EC)*(dps)**2
            return _T
        return T

    def get_d2Trr(self, phi_ext_0=0):
        def d2Trr(dp, P=np.identity(2)): # dp: dphi/dt
            dp = P.dot(dp)
            (dps, dpr) = (dp[0],dp[1])
            _d2Trr = (1/4.)*(hbar/self.EC)
            return _d2Trr
        return d2Trr

    def get_d2Tsr(self, phi_ext_0=0):
        def d2Tsr(dp, P=np.identity(2)): # dp: dphi/dt
            dp = P.dot(dp)
            (dps, dpr) = (dp[0],dp[1])
            _d2Tsr = -(1/8.)*(hbar/self.EC)
            return _d2Tsr
        return d2Tsr

    def get_d2Tss(self, phi_ext_0=0):
        def d2Tss(dp, P=np.identity(2)): # dp: dphi/dt
            dp = P.dot(dp)
            (dps, dpr) = (dp[0],dp[1])
            _d2Tss = (1/16.)*(hbar/self.EC)*2
            return _d2Tss
        return d2Tss

    def get_HessT(self, phi_ext_0=0):
        def HessT(dp, P=np.identity(2)):
            dp = P.dot(dp)
            (dps, dpr) = (dp[0], dp[1])
            d2Trr_dp = self.get_d2Trr(phi_ext_0=phi_ext_0)([dps, dpr])
            d2Tsr_dp = self.get_d2Tsr(phi_ext_0=phi_ext_0)([dps, dpr])
            d2Tss_dp = self.get_d2Tss(phi_ext_0=phi_ext_0)([dps, dpr])
            _HessT = np.array([[d2Tss_dp, d2Tsr_dp],
                               [d2Tsr_dp, d2Trr_dp]])
            _HessT_basis = np.dot(np.dot(P.T, _HessT), P)
            return _HessT_basis
        return HessT

    def get_quadratic_form(self, A, brute=False):
        x0 = np.array([0, 0])
        res = minimize(A, x0, method='SLSQP', tol=1e-12)
        if res.success:
            if brute is True:
                eps = 1e-6
                offs = A((res.x[0], res.x[1], res.x[2]))
                aa = (A((res.x[0]+eps, res.x[1], res.x[2])) - offs)/eps**2
                bb = (A((res.x[0], res.x[1]+eps, res.x[2])) - offs)/eps**2
                cc = (A((res.x[0], res.x[1], res.x[2]+eps)) - offs)/eps**2
                ab = (A((res.x[0]+eps, res.x[1]+eps, res.x[2])) - offs)/eps**2-aa-bb
                ac = (A((res.x[0]+eps, res.x[1], res.x[2]+eps)) - offs)/eps**2-aa-cc
                bc = (A((res.x[0], res.x[1]+eps, res.x[2]+eps)) - offs)/eps**2-bb-cc
                Hess = np.array([[aa, ab/2, ac/2],
                                 [ab/2, bb, bc/2],
                                 [ac/2, bc/2, cc]])
            else:
                Hess = nd.Hessian(A, step=1e-3)(res.x)/2.
            return res.x, Hess
        else:
            raise Exception
    
    def get_U_matrix(self, parameters=None, mode = 'analytical'):
        if parameters is None:
            parameters=self.varying_params 
        U = self.get_anyU((0,)*self.dim, parameters=parameters)
        if mode == 'analytical':
            x0 = np.zeros(self.dim)
            res = minimize(U, x0, method='SLSQP', tol=1e-12)#, bounds=[(-3*np.pi, 3*np.pi), (-3*np.pi, 3*np.pi)]) ################################################################# becareful bounds
            HessU = self.get_HessnU(2, parameters=parameters)
            quad = res.x, HessU(res.x)/2
        else:
            quad = self.get_quadratic_form(U) # not suported anymore
        return quad
    
    def get_T_matrix(self, parameters=None, mode = 'analytical'):
        if parameters is None:
            parameters=self.varying_params 
        phi_ext_0 = parameters['phi_ext_0']
        T = self.get_T(phi_ext_0=phi_ext_0)
        if mode == 'analytical':
            res = np.array([0, 0])
            HessT = self.get_HessT(phi_ext_0=phi_ext_0)
            quad = res, HessT([res[0], res[1]])/2
        else:
            quad = self.get_quadratic_form(T)
        return quad

    def get_freqs_kerrs(self, parameters):
        if parameters is None:
            parameters=self.varying_params 
        res = self.get_normal_mode_frame(parameters=parameters)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi

        # calculate Kerrs from polynomial approximation of potential

        Hess2U = self.get_HessnU(2, parameters=parameters)
        Hess3U = self.get_HessnU(3, parameters=parameters)
        Hess4U = self.get_HessnU(4, parameters=parameters)

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


#        Xi22 = 4 * popt22[2]*(ZPF[2]**2)*(ZPF[3]**2)/2/np.pi

        # Plot to check fits to polynomial expansion
#        fig, ax = plt.subplots()
#        ax.plot(xVec, UxVec, label='UxVec')
#        ax.plot(xVec, popt_x[-1]+popt_x[-2]*xVec+popt_x[-3]*xVec**2,
#                label='fit2')
#        ax.plot(xVec, UxVec - (popt_x[-1]+popt_x[-2]*xVec+popt_x[-3]*xVec**2))
#        ax.plot(xVec, popt_x[-4]*xVec**3, label='fit3')
#        ax.plot(xVec, popt_x[-4]*xVec**3+popt_x[-5]*xVec**4, label='fit4')
#        ax.legend()
        return res1, res2, Xi2, Xi3, Xi4, check_Xi2

    def get_freqs_only(self, parameters):
        if parameters is None:
            parameters=self.varying_params 
        res = self.get_normal_mode_frame(parameters=parameters)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi
        return fs

    def get_normal_mode_frame(self, parameters=None):
        if parameters is None:
            parameters=self.varying_params 
        res1, U0 = self.get_U_matrix(parameters=parameters, mode = 'analytical')
        res2, T0 = self.get_T_matrix(parameters=parameters, mode = 'analytical')
        
        w0, v0 = nl.eigh(T0)
        sqrtw = sl.sqrtm(np.diag(w0))
        # print(nl.norm(np.dot(np.dot(v0, np.diag(w0)), nl.inv(v0))-T0) \
        # /nl.norm(T0))
        U1 = np.dot(np.dot(nl.inv(v0), U0), v0)
        U2 = np.dot(np.dot(nl.inv(sqrtw), U1), nl.inv(sqrtw))
        w2, v2 = nl.eigh(U2)
        P = np.dot(np.dot(v0, nl.inv(sqrtw)), v2)
        invP = np.dot(np.dot(nl.inv(v2), nl.inv(sqrtw)), nl.inv(v0))

        pseudo_invP = np.dot(np.dot(nl.inv(v2), np.diag(w0**0.5)), nl.inv(v0))
        
        wU3 = np.diag(np.dot(np.dot(invP, U0), P))

        return res1, res2, P, wU3
