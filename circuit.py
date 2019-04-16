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
from scipy.optimize import minimize, least_squares, fsolve
from scipy.interpolate import splrep, sproot, splev
from scipy.misc import derivative, factorial
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

def get_factor(which):
    factor = 1
    for elt in set(which):
        factor = factor*factorial(which.count(elt))
    return factor

def permute(to_add, old=[]):
    new = []
    for _to_add in to_add:
        new_list = old+[_to_add]
        new_to_add = to_add.copy()
        new_to_add.remove(_to_add)
        if len(new_to_add)>=1:
            permuted = permute(new_to_add, new_list)
            for perm in permuted:
                new.append(perm)
        else:
            new.append(new_list)
    return new

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
        self.to_compare_0 = None
        self.to_compare_2 = None
        self.permutations = None
        
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
        print('Detected U variables : '+str(self.U_variables))
        self.dim = len(self.U_variables)

        U_expr_sub = U_expr.subs(self.__dict__)
        self.U_formal = U_expr_sub
        
    def prepare_T_formal(self): 
        
        T_expr = parse_expr(self.T_str, evaluate=False)
        print('T = '+str(T_expr))
        T_expr_symbols = get_symbol_list(T_expr)
        self.T_variables = self.remove_params(T_expr_symbols)
        print('Detected T variables : '+str(self.T_variables))
        self.dim = len(self.T_variables)

        T_expr_sub = T_expr.subs(self.__dict__)
        self.T_formal = T_expr_sub
    
    def get_anyL(self, UorT, which): # which should be a tuple with derivativative wanted (1,0,0) for the first one
        
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
        def _anyL(p, P=np.identity(self.dim)):
            p=P.dot(p)
            return anyLs[which](*p, *parameters)
        
        def anyL(p, P=np.identity(self.dim)):
            if not isinstance(p, np.ndarray):
                raise TypeError('p argument should be a numpy.ndarray')
            else:
                if p.ndim>1:
                    shape = p.shape
                    true_shape = shape[:-1]
                    items_anyL = []
                    reshaped_p = np.reshape(p, (np.array(true_shape).prod(), shape[-1])) # assume p vector is in last axis (may be changed)
                    for item_p in reshaped_p:
                        items_anyL.append(_anyL(item_p, P=P))
                    items_anyL = np.reshape(np.array(items_anyL), true_shape)
                    return items_anyL
                else:     
                    return _anyL(p, P=P)
        return anyL
        
    def get_HessnL(self, UorT, n, **kwargs):
        def _HessnL(p, P=np.identity(self.dim)):
#            p = P.dot(p)   
            _HessnL= []
            for which in which_list(n, self.dim):
                _HessnL.append(self.get_any_precomp_L(UorT, which, **kwargs)(p))
            shape=[self.dim for ii in range(n)]
            _HessnL=np.array(_HessnL).reshape(shape)
            
            permutations = tuple_list(n)
            for permutation in permutations:
#                print(P.shape, _HessnL.shape)
                _HessnL = np.transpose(np.dot(np.transpose(_HessnL, permutation), P), permutation)
#            _Hess4U1 = np.transpose(np.dot(np.transpose(_Hess4U,(0,1,2,3)), P),(0,1,2,3))
#            _Hess4U2 = np.transpose(np.dot(np.transpose(_Hess4U1,(1,0,2,3)), P),(1,0,2,3))
#            _Hess4U3 = np.transpose(np.dot(np.transpose(_Hess4U2,(2,1,0,3)), P),(2,1,0,3))
#            _Hess4U4 = np.transpose(np.dot(np.transpose(_Hess4U3,(3,1,2,0)), P),(3,1,2,0))
            return _HessnL
        
        def HessnL(p, P=np.identity(self.dim)):
            if not isinstance(p, np.ndarray):
                raise TypeError('p argument should be a numpy.ndarray')
            else:
                if p.ndim>1:
                    shape = p.shape
                    true_shape = shape[:-1]
                    items_HessnL = []
                    reshaped_p = np.reshape(p, (np.array(true_shape).prod(), shape[-1])) # assume p vector is in last axis (may be changed)
                    for item_p in reshaped_p:
                        items_HessnL.append(_HessnL(item_p, P=P))
                    shape_Hess = items_HessnL[-1].shape
                    items_HessnL = np.reshape(np.array(items_HessnL), true_shape+shape_Hess)
                    return items_HessnL
                else:     
                    return _HessnL(p, P=P)

        return HessnL


    def get_quadratic_form(self, A, brute=False):
        x0 = np.zeros(self.dim)
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
            
    def get_U_matrix(self, mode = 'analytical', search = 'numerical', **kwargs):
        search='global'
        U = self.get_any_precomp_L('U', (0,)*self.dim, **kwargs)
#        print(U([1,2]))
        if mode == 'analytical':
            if search=='analytical':
                def display_f(p):
                    which = np.eye(self.dim, dtype=int)
                    product_deriv = self.get_any_precomp_L('U', tuple(which[0]), **kwargs)(p)
                    for w in which[1:]:
                        product_deriv += self.get_any_precomp_L('U', tuple(w), **kwargs)(p)
                    return product_deriv
                def derivative(p):
                    which = np.eye(self.dim, dtype=int)
                    deriv = []
                    for w in which:
                        deriv.append(self.get_any_precomp_L('U', tuple(w), **kwargs)(p))
                    return deriv
                
                def pos_2nd_derivative(p):
                    which = np.eye(self.dim, dtype=int)*2
                    is_it = True
                    for w in which:
                        is_it = (self.get_any_precomp_L('U', tuple(w), **kwargs)(p)>0) and is_it
                    return is_it
                
                bound = 3*pi
                N = 21
                phi_test_opt = np.linspace(-bound, bound, N)
                phi_test_zero = np.zeros(N)
                phi_test = np.concatenate((np.array([phi_test_zero]), np.array([phi_test_opt]))).T
                
                pot = derivative(phi_test)[1]

                spline = splrep(phi_test_opt, pot)
                def interpol(x):
                    return splev(x, spline)

                phi_interpol = np.linspace(-bound, bound, 10*N)
                
                roots = np.array(sproot(spline))

                is_it = []
                for root in roots:
                    sgn = pos_2nd_derivative(np.array([0,root]))
                    is_it.append(sgn)
                min_roots = roots[np.array(is_it)]
                if len(min_roots)>1 and False:
                    fig0, ax0 = plt.subplots()
#                    ax0.plot(phi_test_opt, derivative(phi_test)[1], '.')
#                    ax0.plot(phi_interpol, interpol(phi_interpol))
#                    ax0.plot(roots, interpol(roots), 'o') 
#                    ax0.plot(min_roots, interpol(min_roots), 'o')   
                    phi_plot = np.concatenate((np.array([np.zeros(N*10)]), np.array([phi_interpol]))).T    
                    
                    U_roots = U(np.concatenate((np.array([np.zeros(len(roots))]), np.array([roots]))).T)
                    U_min_roots = U(np.concatenate((np.array([np.zeros(len(min_roots))]), np.array([min_roots]))).T)
                    ax0.plot(phi_plot, U(phi_plot))
                    ax0.plot(roots, U_roots, 'o') 
                    ax0.plot(min_roots, U_min_roots, 'o')

                
#                Ntest = 101
#                phi_test1b = np.linspace(-3*3*pi, 3*3*pi, Ntest)
#                phi_test1a = np.linspace(0, 0, Ntest)
#                grid1 = np.moveaxis(np.meshgrid(phi_test1a, phi_test1b),0,-1)
#                U_pcolor1 = U(grid1)
#                
#                phi_test2b = np.linspace(0, 0, Ntest)
#                phi_test2a = np.linspace(-2, 2, Ntest)
#                grid2 = np.moveaxis(np.meshgrid(phi_test2a, phi_test2b),0,-1)
#                U_pcolor2 = U(grid2)
#                
#                phi_test3b = np.linspace(-10*pi, 10*pi, Ntest)
#                phi_test3a = np.linspace(-2, 2, Ntest)
#                grid3 = np.moveaxis(np.meshgrid(phi_test3a, phi_test3b),0,-1)
#                U_pcolor3 = U(grid3)
#                
#                roots = fsolve(f, [1 for ii in range(self.dim)])
#                print(roots)
#                
#                f_pcolor = np.abs(display_f(grid3))
#                fig, ax = plt.subplots(3,2)
#                ax[0, 0].pcolor(phi_test3a, phi_test3b, U_pcolor1)
#                ax[1, 0].pcolor(phi_test3a, phi_test3b, U_pcolor2)
#                ax[2, 0].pcolor(phi_test3a, phi_test3b, U_pcolor3)
#                                
#                ax[2, 1].pcolor(phi_test3a, phi_test3b, f_pcolor)
                x0 = np.zeros(self.dim)
                x0 = np.dstack((np.zeros(len(min_roots)), np.array(min_roots)))[0]
                pot_x0 = U(x0)
                pot_x0, x0 = zip(*sorted(zip(pot_x0, x0)))
                x0 = np.array(x0)
#                print(type(x0), type(x0[0]))
#                x0 = np.array([0,min_roots[0]])
#                print(x0)
            if search=='global':
                Ntest = 101
                phi_test = np.linspace(-2*pi, 2*pi, Ntest)
                #TODO reput former line
#                grid = np.meshgrid(*([phi_test]*self.dim))
                grid = np.meshgrid(phi_test, [0])
                grid = np.moveaxis(grid,0,-1)
#                grid = np.reshape(grid, (Ntest**self.dim,self.dim))
                grid = np.reshape(grid, (Ntest,self.dim))
                U_min = U(grid[0])
                ii_min = 0
                for ii in range(1,len(grid)):
                    if U(grid[ii]) < U_min:
                        U_min = U(grid[ii])
                        ii_min = ii
#                ii_0 = np.unravel_index(ii_min, tuple([Ntest]*self.dim))
                ii_0 = np.unravel_index(ii_min, tuple([Ntest,1]))
                x0 = np.array([[phi_test[i] for i in ii_0]])
                print('Global minimum approximate location')
                print(ii_0)
            if search=='numerical':
                x0 = np.zeros(self.dim)
                def U1(x):
                    return U(x)/1e14
#                print(U1(np.array([0,0])))
                res = minimize(U1, x0, method='SLSQP', tol=1e-12)#, bounds=[(-3*np.pi, 3*np.pi), (-3*np.pi, 3*np.pi)]) ################################################################# becareful bounds
                x0 = np.array([res.x])
            HessU = self.get_HessnL('U', 2, **kwargs)
            quad = x0, HessU(x0)

        else:
            quad = self.get_quadratic_form(U) # not suported anymore
        return quad
    
#    def get_U_matrix(self, mode = 'analytical', **kwargs):
#
#        U = self.get_any_precomp_L('U', (0,)*self.dim, **kwargs)
##        print(U([1,2]))
#        if mode == 'analytical':
#            x0 = np.zeros(self.dim)
#            def U1(x):
#                return U(x)/1e14
#            res = minimize(U1, x0, method='SLSQP', tol=1e-12)#, bounds=[(-3*np.pi, 3*np.pi), (-3*np.pi, 3*np.pi)]) ################################################################# becareful bounds
#            HessU = self.get_HessnL('U', 2, **kwargs)
#            quad = res.x, HessU(res.x)/2
##            print(quad)
##            print(res.x)
#        else:
#            quad = self.get_quadratic_form(U) # not suported anymore
#        return quad
    
    def get_T_matrix(self, mode = 'analytical', **kwargs):

        T = self.get_any_precomp_L('T', (0,)*self.dim, **kwargs)
#        print(T([1,2]))
        if mode == 'analytical':
            res = np.zeros(self.dim)
            HessT = self.get_HessnL('T', 2, **kwargs)
            quad = res, HessT(res)
        else:
            quad = self.get_quadratic_form(T)
        return quad

    def get_freqs_kerrs(self, particulars=None, return_components=False, max_solutions=1, sort=False, **kwargs): #particulars should be list of tuple
        res = self.get_normal_mode_frame(sort=False, **kwargs)
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
            
            popt2 = 2*np.array([Hess2_r[ii, ii]/2 for ii in range(self.dim)]) # 1/2*U_matrix**2 ie un w/2 vient de u et un w/2 vient de T
            popt3 = np.array([Hess3_r[ii, ii, ii]/6 for ii in range(self.dim)])
            popt4 = np.array([Hess4_r[ii, ii, ii, ii]/24 for ii in range(self.dim)])
            
#            print('popt2') 
#            print(popt2) # should be omega/2 1/2*w/2*phi**2 for linear part of potential energy
            if particulars is not None:
                Xip = []
                for particular in particulars:
                    factor = get_factor(particular)
                    if len(particular)==2:
                        poptp = Hess2_r[particular]/factor
                        Xip.append(poptp/2/np.pi)
                    elif len(particular)==3:
                        poptp = Hess3_r[particular]/factor
                        Xip.append(poptp/2/np.pi)
                    elif len(particular)==4:
                        poptp = Hess4_r[particular]/factor
                        Xip.append(poptp/2/np.pi)
                Xip = np.array(Xip)
            else:
                Xip = None
                
            # factor 2 see former remark so in front of phi**2 we got w/4 (one 2 come from developping (a^+ + a)**2, the other from the kinetic part)
            Xi2 = 2 * popt2/2/np.pi # freq en Hz : coeff devant a^+.a (*2 to get whole freq)
            Xi3 = 3 * popt3/2/np.pi #coeff devant a^2.a^+
            Xi4 = 6 * popt4/2/np.pi #coeff devant a^2.a^+2
            
            Xi2s.append(Xi2)
            Xi3s.append(Xi3)
            Xi4s.append(Xi4)
            Xips.append(Xip)
    
        n_solutions = len(Xi2s)
        if n_solutions<max_solutions:
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
            return res1s, res2, Xi2s, Xi3s, Xi4s, Xips, None

    def get_freqs_only(self, sort=True, **kwargs):
        res = self.get_normal_mode_frame(sort=sort,**kwargs)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi
        return fs

    def get_normal_mode_frame(self, sort=True, **kwargs):
        res1s, U0s = self.get_U_matrix(mode = 'analytical', **kwargs)
        res2, T0 = self.get_T_matrix(mode = 'analytical', **kwargs)
        
        
        Ps = []
        w2s = []
        for U0 in U0s:
            w0, v0 = nl.eigh(T0)
    #        w0, v0 = self.reorder(*(nl.eigh(T0)), 0, debug=False)
            sqrtw = sl.sqrtm(np.diag(w0))
            U1 = np.dot(np.dot(nl.inv(v0), U0), v0)
            T1 = np.dot(np.dot(nl.inv(v0), T0), v0)
            U2 = np.dot(np.dot(nl.inv(sqrtw), U1), nl.inv(sqrtw))
            T2 = np.dot(np.dot(nl.inv(sqrtw), T1), nl.inv(sqrtw))
            w2, v2 = nl.eigh(U2)
#            T2 = identité
#            U2 = matric qcq
#            w2 = list omega**2
            
    #        w2, v2 = self.reorder(w2, v2, 2, debug=False)
    
    
            P = np.dot(np.dot(v0, nl.inv(sqrtw)), v2)
    #        print(w0)
            if sort:
                w2, P = self.reorder(w2, P, 2)
    
            tP = np.dot(np.dot(v2.T, nl.inv(sqrtw)), v0.T)
    
            
            wT3 = np.dot(np.dot(tP, T0), P)
            wU3 = np.dot(np.dot(tP, U0), P)

            
            phiZPF = np.sqrt(1/2/np.sqrt(w2))
            wU3 = np.diag(phiZPF)*wU3*np.diag(phiZPF)
            P = np.dot(np.diag(phiZPF), P)
            tP = np.dot(tP, np.diag(phiZPF))
            wT3 = np.dot(np.dot(tP, T0), P)
            wU3 =np.dot(np.dot(tP, U0), P)
            
#            print('T&U')
#            print(wT3)
#            print(wU3)

            Ps.append(P)
            w2s.append(w2) # differents minima
#            self.print_P(P)

        return res1s, res2, np.array(Ps), np.array(w2s)
    
    def print_P(self, P):
        print('\n### P ###')
        p = self.U_variables
        for ii, p_j in enumerate(p):
            to_print = p_j + ' = '
            to_add = ''
            for jj in range(self.dim):
                to_add = to_add+'%.5f'%P[ii,jj]+'*p%d + '%(jj)
            to_print += to_add[:-3]
            print(to_print)
        print('   with :')
        for jj in range(self.dim):
            print('   p%d = a%d + a%d^+'%(jj, jj, jj))
        print('#########\n')
    
    def reorder(self, values, vectors, either0_or2, debug=False):
        dim = len(values)
        vectors = vectors.T
        order = []
        if either0_or2 ==2:
#            print(vectors)
            val, vec = nl.eigh(vectors.T)
#            print('val propres 0 =' +str(val))
        
#        print(self.to_compare_0, self.to_compare_2)
        
        to_compares = [self.to_compare_0, self.to_compare_2]
        jj = int(either0_or2/2)
        if to_compares[jj] is None:
            to_compare = np.eye(dim)
            self.permutations = permute([ii for ii in range(dim)])
        else:
            to_compare = np.eye(dim)
#            to_compare = to_compares[jj]
            
        distances = []
        sgns = []
        for ii, vector in enumerate(vectors):
            if debug:
                print('comparison')
                print(np.dot(to_compare, vector))
            dists, sgn = comp_dist(vector, to_compare)
            distances.append(dists)
            sgns.append(sgn)
#            if either0_or2==2:
#                print(dists)
            index_max = np.argmin(dists)
#            vectors[ii] = vector*sgn
            order.append(index_max)
        
        best_sum_dist = np.inf
        best_perm = None
        for perm in self.permutations:
            sum_dist = 0
            for ii in perm:
                sum_dist += distances[ii][perm[ii]]
            if sum_dist<best_sum_dist:
                best_perm = perm
                best_sum_dist = sum_dist
                
        order=best_perm
#        order=[0,1,2,3,4]
        for ii in range(dim):
            vectors[ii]=vectors[ii]*sgns[ii][order[ii]]
#        if either0_or2==2:
#            print(order)
        
#        print(order)
        
#        order_bis = [[] for i in range(dim)]
#        for ii, vectorT in enumerate(vectors.T):
#            order_bis[np.argmax(np.abs(np.dot(to_compare, vectorT)))].append(ii)
#        print(order)
#        print(order_bis)
#        if to_compares[jj] is None:
#            to_correct = None
#            for ii in range(dim):
#                if len(order_bis[ii])==2:
#                    order_bis[ii].remove(order[ii])
#                    to_correct = order_bis[ii][0]
#                elif len(order_bis[ii])==3:
#                    print('Houston we got a problem')
#            if to_correct is not None:
#                for ii in range(dim):
#                    if len(order_bis[ii])==0:
#                        order[ii]=to_correct
    
    #    order_temp = list(range(dim))
    #    pb_index = None
    #    
    #    indices = [[] for i in range(dim)]
    #    for ii, elt in enumerate(order):
    #        indices[elt].append(ii)
    #    
    #    
    #    for ii, elt in enumerate(order):
    #        if ii==0:
    #            order_temp.remove(elt)
    #        else:
    #            if elt not in order[:ii]:
    #                order_temp.remove(elt)
    #            else:
    #                pb_index = ii
    #                
    #    if pb_index is not None:
    #        order[ii]=order_temp[0]
        
        vectors_ordered = [0 for i in range(dim)]         
        values_ordered = [0 for i in range(dim)]
        
        for ii in range(dim):
            vectors_ordered[order[ii]] = vectors[ii]
            values_ordered[order[ii]] = values[ii]
        
                
        if to_compares[jj] is None:
            if either0_or2==0:
                self.to_compare_0 = np.array(vectors_ordered)
            elif either0_or2==2:
                self.to_compare_2 = np.array(vectors_ordered)
        if debug:
            print(order)
            print(vectors)
            print(vectors_ordered)
            
        vectors_ordered = np.array(vectors_ordered).T 
        values_ordered = np.array(values_ordered)
    #    print(order)
        return values_ordered, vectors_ordered
    
def comp_dist(a, b):
    ret=[]
    sgn=[]
    for ii in b:
        dists = np.array([(((a-ii)**2).sum())**0.5, (((a+ii)**2).sum())**0.5])
        argmin = np.argmin(dists)
        ret.append(dists[argmin])
        sgn.append(argmin*(-2)+1)
    return np.array(ret), np.array(sgn)


def pcolor_z(ax, *args, alpha=None, norm=None, cmap=None, vmin=None, vmax=None, data=None, **kwargs):
    ax.pcolor(*args, alpha=None, norm=None, cmap=cmap, vmin=vmin, vmax=vmax, data=data, **kwargs)
    if len(args)==3:
        x_axis, y_axis, z_data = args
    elif len(args)==1:
        z_data, = args
        shape_data = z_data.shape
        x_axis, y_axis = np.arange(shape_data[1]), np.arange(shape_data[0])
    else:
        raise ValueError('Should have x, y, z or z args')
    def format_coord(x, y):
        dx = (x_axis[1]-x_axis[0])
        dy = (y_axis[1]-y_axis[0])
        col = np.argmin(np.abs(x_axis-x+dx/2))
        row = np.argmin(np.abs(y_axis-y+dy/2))
        numrows, numcols = np.shape(z_data)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = z_data[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)
    ax.format_coord = format_coord
    
def to_pcolor(x, y):
    if len(x)>1:       
        xf = 2*x[-1]-x[-2]
    else:
        xf = x[0]+1
    if len(y)>1:
        yf = 2*y[-1]-y[-2]
    else:
        yf = y[0]+1
    x = np.append(x, xf)
    y = np.append(y, yf)
    return (x, y)