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
import numdifftools as nd
from scipy.misc import derivative
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


Phi0 = sc.value('mag. flux quantum')
e = sc.elementary_charge
phi0 = Phi0/2/np.pi  # Phi_0=h/(2*e)
pi = sc.pi
hbar = sc.hbar
h = sc.h

fs = 25
ls = 25


def cosm(A):
    return ((1j*A).expm()+(-1j*A).expm())/2


def sinm(A):
    return ((1j*A).expm()-(-1j*A).expm())/2j

def convert_EJ_LJ_I0(EJ=None, LJ=None, I0=None):
    if EJ != None:
        _EJ = EJ
        _LJ = phi0**2/EJ
        _I0 = EJ/phi0
    if LJ != None:
        _EJ = phi0**2/LJ
        _LJ = LJ
        _I0 = _EJ/phi0
    if I0 != None:
        _EJ = I0*phi0
        _LJ = phi0**2/_EJ
        _I0 = I0
    return(_EJ, _LJ, _I0)

def get_w_Z_LJ_from_E(EC, EL, EJ):
    C = e**2/2/EC
    L = phi0**2/EL
    LJ = phi0**2/EJ
    w = 1/np.sqrt(L*C)
    Z = np.sqrt(L/C)
    return w, Z, LJ


def get_E_from_w(w, Z, LJ):
    C = 1/(Z*w)
    L = Z/w
    EL = phi0**2/L
    EC = e**2/2/C
    EJ = phi0**2/LJ
    return EC, EL, EJ


def get_phi_EJ_from_Z_LJ(Za, Zb, LJ):
    phia = (1/phi0) * (np.sqrt((hbar/2)*Za))  # phiZPF
    phib = (1/phi0) * (np.sqrt((hbar/2)*Zb))  # phiZPF
    EJoverhbar = phi0**2/LJ/hbar
    return phia, phib, EJoverhbar

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
    if n==0:
        return [(0,)*dim]
    elif n==1:
        loc_list = []
        for ii in range(dim):
            loc_list.append(tuple([0 if jj!=ii else 1 for jj in range(dim)]))
        return loc_list
    else:
        loc_list = which_list(1, dim)
        new_which_list = []
        for which in which_list(n-1, dim):
            for loc in loc_list:
                new_which_list.append(tuple(np.array(which)+np.array(loc)))
        return new_which_list
    
def tuple_list(n):
    _tuple_list = [tuple(range(n))]
    for ii in range(n-1):
        temp_list = list(range(n))
        temp_list[n-ii-2]=n-1
        temp_list[n-1]=n-ii-2
        _tuple_list.append(tuple(temp_list))
    return _tuple_list


class Circuit(object):

    def __init__(self):
        self.hbar = hbar
        self.pi = pi

        
        self.anyUs={}
        self.anyTs={}
        self.prepare_U_formal()
        self.prepare_T_formal()
        self.store_anyL('U', self.max_order)
        self.store_anyL('T', self.max_order)
        
    def remove_params(self, symbol_list):
        variables = []
        for symbol in symbol_list:
            if (not symbol in self.__dict__.keys()) and (not symbol in self.varying_params.keys()):
                variables.append(symbol)
        variables = sorted(variables)
        return variables
    
    def find_parameters(self, kwargs):
        for varying_param in self.varying_params.keys():
            if varying_param not in kwargs.keys():
                kwargs[varying_param] = self.varying_params[varying_param] # set to default value if not given by user 
        parameters=[]
        for key in sorted(kwargs.keys()):
            parameters.append(kwargs[key])
        return parameters
    
    
    def prepare_U_formal(self): 
        U_expr = parse_expr(self.U_str, evaluate=False)
        print('U = '+str(U_expr))
        U_expr_symbols = get_symbol_list(U_expr)
        self.U_variables = self.remove_params(U_expr_symbols)
        self.dim = len(self.U_variables)

        U_expr_sub = U_expr.subs(self.__dict__)
        self.U_formal = U_expr_sub
        
    def prepare_T_formal(self): 
        
        T_expr = parse_expr(self.T_str, evaluate=False)
        print('T = '+str(T_expr))
        T_expr_symbols = get_symbol_list(T_expr)
        self.T_variables = self.remove_params(T_expr_symbols)
        self.dim = len(self.T_variables)

        T_expr_sub = T_expr.subs(self.__dict__)
        self.T_formal = T_expr_sub
    
    def get_anyL(self, UorT, which): # which should be a tuple with derivativative wanted (0,0) for the first one
        
        if UorT=='U':
            L_expr = self.U_formal
            L_variables = self.U_variables
        if UorT=='T':
            L_expr = self.T_formal
            L_variables = self.T_variables
        if sum(which)==0:
            _anyL = L_expr
        else:
            which_format = format_diff(which, L_variables)
            _anyL = sp.diff(L_expr, *which_format)
        anyL = sp.lambdify(tuple(L_variables)+tuple(sorted(self.varying_params.keys())), _anyL, 'numpy')
        return anyL
    
    def store_anyL(self, UorT, up_to_n):
        if UorT=='U':
            anyLs = self.anyUs
        if UorT=='T':
            anyLs = self.anyTs
        for nn in range(up_to_n+1):
            for which in which_list(nn, self.dim):
                anyLs[which]=self.get_anyL(UorT, which)
    
    def get_any_precomp_L(self, UorT, which, **kwargs):
        if UorT=='U':
            anyLs = self.anyUs
        if UorT=='T':
            anyLs = self.anyTs
            
        parameters = self.find_parameters(kwargs)
        def anyL(p, P=np.identity(self.dim)):
            p=P.dot(p)
            return anyLs[which](*p, *parameters)
        return anyL
        
    def get_HessnL(self, UorT, n, **kwargs):
        def HessnL(p, P=np.identity(self.dim)):
#            p = P.dot(p)   
            _HessnL= []
            for which in which_list(n, self.dim):
                _HessnL.append(self.get_any_precomp_L(UorT, which, **kwargs)(p))
            shape=[self.dim for ii in range(n)]
            _HessnL=np.array(_HessnL).reshape(shape)
            
            permutations = tuple_list(n)
            for permutation in permutations:
                _HessnL = np.transpose(np.dot(np.transpose(_HessnL, permutation), P), permutation)
#            _Hess4U1 = np.transpose(np.dot(np.transpose(_Hess4U,(0,1,2,3)), P),(0,1,2,3))
#            _Hess4U2 = np.transpose(np.dot(np.transpose(_Hess4U1,(1,0,2,3)), P),(1,0,2,3))
#            _Hess4U3 = np.transpose(np.dot(np.transpose(_Hess4U2,(2,1,0,3)), P),(2,1,0,3))
#            _Hess4U4 = np.transpose(np.dot(np.transpose(_Hess4U3,(3,1,2,0)), P),(3,1,2,0))
            return _HessnL
        return HessnL


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
    
    def get_U_matrix(self, mode = 'analytical', globalsearch = True, **kwargs):

        U = self.get_any_precomp_L('U', (0,)*self.dim, **kwargs)
#        print(U([1,2]))
        if mode == 'analytical':
            if globalsearch is True:
                Ntest = 101
                phi_test = np.linspace(-10*pi, 10*pi, Ntest)
                grid = np.meshgrid(*([phi_test]*self.dim))
                grid = np.moveaxis(grid,0,-1)
                grid = np.reshape(grid, (Ntest**self.dim,self.dim))
                U_min = U(grid[0])
                ii_min = 0
                for ii in range(1,len(grid)):
                    if U(grid[ii]) < U_min:
                        U_min = U(grid[ii])
                        ii_min = ii
                ii_0 = np.unravel_index(ii_min, tuple([Ntest]*self.dim))
                x0 = [phi_test[i] for i in ii_0]
                print('Global minimum approximate location')
                print(ii_0)
            else:
                x0 = np.zeros(self.dim)
            def U1(x):
                return U(x)/1e14
            res = minimize(U1, x0, method='SLSQP', tol=1e-12)#, bounds=[(-3*np.pi, 3*np.pi), (-3*np.pi, 3*np.pi)]) ################################################################# becareful bounds
            HessU = self.get_HessnL('U', 2, **kwargs)
            quad = res.x, HessU(res.x)/2
    #            print(quad)
    #            print(res.x)
        else:
            quad = self.get_quadratic_form(U) # not suported anymore
        return quad
    
    def get_T_matrix(self, mode = 'analytical', **kwargs):

        T = self.get_any_precomp_L('T', (0,)*self.dim, **kwargs)
#        print(T([1,2]))
        if mode == 'analytical':
            res = np.array([0, 0])
            HessT = self.get_HessnL('T', 2, **kwargs)
            quad = res, HessT(res)/2
        else:
            quad = self.get_quadratic_form(T)
        return quad

    def get_freqs_kerrs(self, **kwargs):

        res = self.get_normal_mode_frame(**kwargs)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi

        # calculate Kerrs from polynomial approximation of potential

        Hess2U = self.get_HessnL('U', 2,  **kwargs)
        Hess3U = self.get_HessnL('U', 3,  **kwargs)
        Hess4U = self.get_HessnL('U', 4,  **kwargs)

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

        return res1, res2, Xi2, Xi3, Xi4, check_Xi2

    def get_freqs_only(self,  **kwargs):
        res = self.get_normal_mode_frame(**kwargs)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi
        return fs

    def get_normal_mode_frame(self, **kwargs):
        res1, U0 = self.get_U_matrix(mode = 'analytical', **kwargs)
        res2, T0 = self.get_T_matrix(mode = 'analytical', **kwargs)
        
        w0, v0 = nl.eigh(T0)
        sqrtw = sl.sqrtm(np.diag(w0))

        U1 = np.dot(np.dot(nl.inv(v0), U0), v0)
        U2 = np.dot(np.dot(nl.inv(sqrtw), U1), nl.inv(sqrtw))
        w2, v2 = nl.eigh(U2)
        P = np.dot(np.dot(v0, nl.inv(sqrtw)), v2)
        invP = np.dot(np.dot(nl.inv(v2), nl.inv(sqrtw)), nl.inv(v0))

        pseudo_invP = np.dot(np.dot(nl.inv(v2), np.diag(w0**0.5)), nl.inv(v0))
        
        wU3 = np.diag(np.dot(np.dot(invP, U0), P))

        return res1, res2, P, wU3
