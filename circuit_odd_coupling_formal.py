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

class CircuitOddCoupling(Circuit):

    def __init__(self, ECa, ELa, ECb, ELb, EJ1, EJ2,
                 ECc=None, ELc=None, Ecoup=None,
                 printParams=True):
        self.hbar = hbar
        
        self.varying_params={'phi_ext_s_0':0, 'phi_ext_l_0':0}

        # from http://arxiv.org/abs/1602.01793
        wa, Za, LJ = get_w_Z_LJ_from_E(ECa, ELa, EJ1)
        wb, Zb, LJ = get_w_Z_LJ_from_E(ECb, ELb, EJ1)
        phia = (1/phi0) * (np.sqrt((hbar/2)*Za))  # phiZPF
        phib = (1/phi0) * (np.sqrt((hbar/2)*Zb))  # phiZPF
        na_zpf = (1/(2*e)) * (np.sqrt(hbar/2/Za))  # nZPF
        nb_zpf = (1/(2*e)) * (np.sqrt(hbar/2/Zb))  # nZPF
        Ca = 1/(Za*wa)
        Cb = 1/(Zb*wb)
        La = Za/wa
        Lb = Zb/wb
        self.ELa = phi0**2/La
        self.ELb = phi0**2/Lb
        self.ELc = ELc
        self.ECa = e**2/2/Ca
        self.ECb = e**2/2/Cb
        self.ECc = ECc
        self.EJ1 = EJ1
        self.EJ2 = EJ2
        self.EJ = (EJ1+EJ2)/2
        self.dEJ = (EJ1-EJ2)/2
        self.Ecoup = Ecoup
        epsilonbar = pi/2/3
        g4 = (self.EJ/2)*epsilonbar*(1./math.factorial(4))*phia**4*phib
        g2 = (self.EJ/2)*epsilonbar*(1./math.factorial(2))*phia**2*phib

        hbarXiaa = 0.5*(self.EJ/100.)*phia**4  # 1 % error on EJ
        hbarXiab = (self.EJ/100.)*phia**2*phib**2

        omega_plasma = 2*pi*24e9
        CJ = 1/(omega_plasma**2*(2*LJ))  # each junction has 2*LJ
        beta = 2*CJ/np.sqrt((Ca+CJ)*(Cb+CJ))  # 2.104 in Steve's notes
        g = beta*np.sqrt(wa*wb)  # energy/hbar 2.110 in Steve's notes
        kappaa_over_kappab = g**2/(wa-wb)**2  # 6.40
        #  Phia = phia*qt.tensor(a+a.dag(), qt.qeye(nb))
        #  Phib = phib*qt.tensor(qt.qeye(na), b+b.dag())

        if printParams:
            print("La = "+str(La*1e9)+" nH")
            print("Ca = "+str(Ca*1e15)+" fF")
            print("Lb = "+str(Lb*1e9)+" nH")
            print("Cb = "+str(Cb*1e15)+" fF")
            print("ELa/h = "+str(1e-9*self.ELa/hbar/2/pi)+" GHz")
            print("ELb/h = "+str(1e-9*self.ELb/hbar/2/pi)+" GHz")
            if self.ELc is not None:
                print("ELc/h = "+str(1e-9*self.ELc/hbar/2/pi)+" GHz")
            print("ECa/h = "+str(1e-9*self.ECa/hbar/2/pi)+" GHz")
            print("ECb/h = "+str(1e-9*self.ECb/hbar/2/pi)+" GHz")
            if self.ECc is not None:
                print("ECc/h = "+str(1e-9*self.ECc/hbar/2/pi)+" GHz")
            print("EJ/h = "+str(1e-9*self.EJ/hbar/2/pi)+" GHz")
            if self.Ecoup is not None:
                print("Ecoup/h = "+str(1e-9*self.Ecoup/hbar/2/pi)+" GHz")
            print("phia_zpf = "+str(phia))
            print("phib_zpf = "+str(phib))
            print("na_zpf = "+str(na_zpf))
            print("nb_zpf = "+str(nb_zpf))
            print("g4/h = "+str(1e-6*g4/hbar/2/pi)+" MHz")
            print("g2/h = "+str(1e-6*g2/hbar/2/pi)+" MHz")
            print("Xiaa/2pi = "+str(1e-6*hbarXiaa/hbar/2/pi)+" MHz")
            print("Xiab/2pi = "+str(1e-6*hbarXiab/hbar/2/pi)+" MHz")
            print("CJ per junction = "+str(CJ*1e15)+str(" fF"))
            print("kappab/kappaa limited by CJ = "+str(1/kappaa_over_kappab))
            
        self.prepare_U_formal()
            
    def remove_params(self, symbol_list):
        variables = []
        for symbol in symbol_list:
            if (not symbol in self.__dict__.keys()) and (not symbol in self.varying_params.keys()):
                variables.append(symbol)
        variables = sorted(variables)
        return variables
    
    def prepare_U_formal(self): 
        U_str = '0.5*(ELa/hbar)*pa**2 + \
                + 0.5*(ELb/hbar)*pb**2 + \
                + 0.5*(ELc/hbar)*pc**2+ \
                - (EJ/hbar)*cos(phi_ext_s_0/2)*cos(pa+pb+phi_ext_s_0/2+phi_ext_l_0) \
                - (dEJ/hbar)*sin(phi_ext_s_0/2) * sin(pa+pb+phi_ext_s_0/2+phi_ext_l_0)'
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
            print(which_format)
            _anyU = sp.diff(U_expr_sub, *which_format)
        print(_anyU)
        _anyU = sp.lambdify(tuple(self.U_variables), _anyU, 'numpy')
        def anyU(p, P=np.identity(self.dim)):
            p=P.dot(p)
            return _anyU(*p)
        return anyU
    
    def get_HessnU(self, n, parameters=None):
        if parameters is None:
            parameters=self.varying_params
        def HessnU(p, P=np.identity(self.dim)):
            p = P.dot(p)
            
            _HessnU= []
            for which in which_list(n, self.dim):
                _HessnU.append(self.get_anyU(which, parameters=parameters))
            shape=[self.dim for ii in range(n)]
            _HessnU=np.array(_HessnU).reshape(shape)
            
            permutations = tuple_list(self.dim)
            for permutation in permutations:
                _HessnU = np.transpose(np.dot(np.transpose(_HessnU,permutation), P),permutation)
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


    def get_T(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def T(dp, P=np.identity(3)): # dp: dphi/dt
            dp = P.dot(dp)
            (dpa, dpb, dpc) = (dp[0],dp[1], dp[2])
            _T = (1/16.)*(hbar/self.ECa)*(dpa)**2 + \
                 (1/16.)*(hbar/self.ECb)*(dpb)**2 + \
                 (1/16.)*(hbar/self.ECc)*(dpc)**2 + \
                 (1/16.)*(hbar/self.Ecoup)*(dpa-dpc)**2
            return _T
        return T
    
    def get_d2Taa(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def d2Taa(dp, P=np.identity(3)):
            dp = P.dot(dp)
            (dpa, dpb, dpc) = (dp[0], dp[1], dp[2])
            _d2Taa = (1/8.)*(hbar/self.ECa)+(1/8.)*(hbar/self.Ecoup)
            return _d2Taa
        return d2Taa
    def get_d2Tbb(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def d2Tbb(dp, P=np.identity(3)):
            dp = P.dot(dp)
            (dpa, dpb, dpc) = (dp[0], dp[1], dp[2])
            _d2Tbb = (1/8.)*(hbar/self.ECb)
            return _d2Tbb
        return d2Tbb
    def get_d2Tcc(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def d2Tcc(dp, P=np.identity(3)):
            dp = P.dot(dp)
            (dpa, dpb, dpc) = (dp[0], dp[1], dp[2])
            _d2Tcc = (1/8.)*(hbar/self.ECc)+(1/8.)*(hbar/self.Ecoup)
            return _d2Tcc
        return d2Tcc
    def get_d2Tac(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def d2Tac(dp, P=np.identity(3)):
            dp = P.dot(dp)
            (dpa, dpb, dpc) = (dp[0], dp[1], dp[2])
            _d2Tac = -(1/8.)*(hbar/self.Ecoup)
            return _d2Tac
        return d2Tac

    def get_HessT(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def HessT(dp, P=np.identity(3)):
            dp = P.dot(dp)
            (dpa, dpb, dpc) = (dp[0], dp[1], dp[2])
            d2Taa_dp = self.get_d2Taa(phi_ext_s_0=phi_ext_s_0,
                                      phi_ext_l_0=phi_ext_l_0)([dpa, dpb, dpc])
            d2Tbb_dp = self.get_d2Tbb(phi_ext_s_0=phi_ext_s_0,
                                      phi_ext_l_0=phi_ext_l_0)([dpa, dpb, dpc])
            d2Tcc_dp = self.get_d2Tcc(phi_ext_s_0=phi_ext_s_0,
                                      phi_ext_l_0=phi_ext_l_0)([dpa, dpb, dpc])
            d2Tab_dp = 0
            d2Tac_dp = self.get_d2Tac(phi_ext_s_0=phi_ext_s_0,
                                      phi_ext_l_0=phi_ext_l_0)([dpa, dpb, dpc])
            d2Tbc_dp = 0
            _HessT = np.array([[d2Taa_dp, d2Tab_dp, d2Tac_dp],
                               [d2Tab_dp, d2Tbb_dp, d2Tbc_dp], 
                               [d2Tac_dp, d2Tbc_dp, d2Tcc_dp]])
            _HessT_basis = np.dot(np.dot(P.T, _HessT), P)
            return _HessT_basis
        return HessT

    def get_quadratic_form(self, A, brute=False):
        x0 = np.array([0, 0, 0])
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
            res = minimize(U, x0, method='SLSQP', tol=1e-12)
            HessU = self.get_HessnU(2, parameters=parameters)
            quad = res.x, HessU([res.x])/2
        else:
            quad = self.get_quadratic_form(U) # not suported anymore
        return quad

    def get_T_matrix(self, parameters=None, mode = 'analytical'):
        if parameters is None:
            parameters=self.varying_params 
        phi_ext_s_0 = parameters['phi_ext_s_0']
        phi_ext_l_0 = parameters['phi_ext_l_0']
        T = self.get_T(phi_ext_s_0=phi_ext_s_0,
                       phi_ext_l_0=phi_ext_l_0)
        if mode == 'analytical':
            res = np.array([0, 0, 0])
            HessT = self.get_HessT(phi_ext_s_0=phi_ext_s_0,
                           phi_ext_l_0=phi_ext_l_0)
            quad = res, HessT([res[0], res[1], res[2]])/2
        else:
            quad = self.get_quadratic_form(T)
        return quad

    def get_freqs_kerrs(self, parameters=None):
        res = self.get_normal_mode_frame(phi_ext_s_0=phi_ext_s_0,
                                         phi_ext_l_0=phi_ext_l_0)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi

        # calculate Kerrs from polynomial approximation of potential
        U = self.get_U(phi_ext_s_0=phi_ext_s_0,
                       phi_ext_l_0=phi_ext_l_0)
        [x0, y0, z0] = (nl.inv(P)).dot([res1[0], res1[1], res1[2]])

        HessU = self.get_HessU(phi_ext_s_0=phi_ext_s_0,
                               phi_ext_l_0=phi_ext_l_0)
        Hess3U = self.get_Hess3U(phi_ext_s_0=phi_ext_s_0,
                               phi_ext_l_0=phi_ext_l_0)
        Hess4U = self.get_Hess4U(phi_ext_s_0=phi_ext_s_0,
                               phi_ext_l_0=phi_ext_l_0)
        Hess_r = HessU([x0, y0, z0], P=P)
#        print('Hess_r = ' + str(Hess_r/2))
#        print('w2 = ' +str(np.diag(Hess_r/2)))
        
        Hess3_r = Hess3U([x0, y0, z0], P=P)       
        Hess4_r = Hess4U([x0, y0, z0], P=P) 
        
        Uxy = lambda x,y : U([x+x0, y+y0, z0], P=P)

        Ux = lambda x: U([x+x0, y0, z0], P=P)
        Uy = lambda y: U([x0, y+y0, z0], P=P)
        Uz = lambda z: U([x0, y0, z+z0], P=P)

        Nx = 5

        xmax = nl.norm((nl.inv(P)).dot(np.array([np.pi/20, 0, 0])))
        xVec = np.linspace(-xmax, xmax, Nx)
        X, Y = np.meshgrid(xVec, xVec, copy=False)
        X = X.flatten()
        Y = Y.flatten()
        poly_form2 = np.array([X*0+1, # 0
              X, Y, # 1, 2
              X**2, 2*X*Y, Y**2, # 3, 4, 5
              X**3, 3*X**2*Y, 3*X*Y**2, Y**3,
              X**4, 4*X**3*Y, 6*X**2*Y**2, 4*X**1*Y**3, Y**4
              ]).T

        UxVec = np.zeros(Nx)
        UyVec = np.zeros(Nx)
        UzVec = np.zeros(Nx)
        UxyVec2 = np.zeros((Nx, Nx))
        for ii, xx in enumerate(xVec):
            UxVec[ii] = Ux(xx)
            UyVec[ii] = Uy(xx)
            UzVec[ii] = Uz(xx)
            for jj, yy in enumerate(xVec):
                UxyVec2[jj, ii] = Uxy(xx, yy)
        Uxyflat2 = UxyVec2.flatten()
        coeff2, r, rank, s = np.linalg.lstsq(poly_form2, Uxyflat2, rcond=-1)
        popt_x = np.polyfit(xVec, UxVec, 4)
        popt_y = np.polyfit(xVec, UyVec, 4)
        popt_z = np.polyfit(xVec, UzVec, 4)

        xmax = nl.norm((nl.inv(P)).dot(np.array([np.pi/4, 0, 0])))
        xVec = np.linspace(-xmax, xmax, Nx)
        X, Y = np.meshgrid(xVec, xVec, copy=False)
        X = X.flatten()
        Y = Y.flatten()
        poly_form2 = np.array([X*0+1, # 0
                               X, Y, # 1, 2
                               X**2, 2*X*Y, Y**2, # 3, 4, 5
                               X**3, 3*X**2*Y, 3*X*Y**2, Y**3,
                               X**4, 4*X**3*Y, 6*X**2*Y**2, 4*X**1*Y**3, Y**4
                               ]).T
        poly_form3 = np.array([X**3, 3*X**2*Y, 3*X*Y**2, Y**3,
                               X**4, 4*X**3*Y, 6*X**2*Y**2, 4*X**1*Y**3, Y**4# 0, 1, 2, 3
                              ]).T
        UxyVec3 = np.zeros((Nx, Nx))
        for ii, xx in enumerate(xVec):
            for jj, yy in enumerate(xVec):
                UxyVec3[jj, ii] = Uxy(xx, yy)
        Uxyflat3 = UxyVec3.flatten() - np.dot(coeff2, poly_form2.T)
        coeff3, r, rank, s = np.linalg.lstsq(poly_form3, Uxyflat3, rcond=-1)

#        fig, ax = plt.subplots(2)
#        ax[0].imshow(UxyVec3)
#        ax[1].imshow(np.reshape(np.dot(coeff2, poly_form2.T), (Nx,Nx)))
#        print(coeff)

#        fx = np.sqrt(derivative(Ux, 0, dx=1e-3)/2)/2/np.pi
#        fy = np.sqrt(nd.Derivative(Uy)(0)/2)/2/np.pi
#        fz = np.sqrt(nd.Derivative(Uz)(0)/2)/2/np.pi
        popt2 = np.array([popt_x[-3], popt_y[-3], Hess_r[0, 0]/2, Hess_r[1, 1]/2, Hess_r[2, 2]/2])
        popt3 = np.array([popt_x[-4], popt_y[-4], Hess3_r[0, 0, 0]/6, Hess3_r[1, 1, 1]/6, Hess3_r[2, 2, 2]/6])
        popt4 = np.array([popt_x[-5], popt_y[-5], Hess4_r[0, 0, 0, 0]/24, Hess4_r[1, 1, 1, 1]/24, Hess4_r[1, 1, 1, 1]/24])
        popt22 = np.array([popt_x[-5], popt_y[-5], Hess4_r[0, 0, 1, 1]/4, Hess4_r[1, 1, 0, 0]/4, Hess4_r[1, 0, 1, 0]/4])

        ZPF = popt2**(-1./4)

        Xi2 = popt2*(ZPF**2)/2/np.pi
        Xi3 = popt3*(ZPF**3)/2/np.pi
        Xi4 = 6 * popt4*(ZPF**4)/2/np.pi
        Xi22 = 4 * popt22[2]*(ZPF[2]**2)*(ZPF[3]**2)/2/np.pi
        # Plot to check fits to polynomial expansion
#        fig, ax = plt.subplots()
#        ax.plot(xVec, UxVec, label='UxVec')
#        ax.plot(xVec, popt_x[-1]+popt_x[-2]*xVec+popt_x[-3]*xVec**2,
#                label='fit2')
#        ax.plot(xVec, UxVec - (popt_x[-1]+popt_x[-2]*xVec+popt_x[-3]*xVec**2))
#        ax.plot(xVec, popt_x[-4]*xVec**3, label='fit3')
#        ax.plot(xVec, popt_x[-4]*xVec**3+popt_x[-5]*xVec**4, label='fit4')
#        ax.legend()
        return res1, res2, fs, Xi2, Xi3, Xi4, coeff2, Xi22
        
    def get_freqs_only(self, phi_ext_s_0=0, phi_ext_l_0=0):
        res = self.get_normal_mode_frame(phi_ext_s_0=phi_ext_s_0,
                                         phi_ext_l_0=phi_ext_l_0)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi
        return fs

    def get_normal_mode_frame(self, phi_ext_s_0=0, phi_ext_l_0=0):
        res1, U0 = self.get_U_matrix(phi_ext_s_0=phi_ext_s_0,
                                     phi_ext_l_0=phi_ext_l_0, mode = 'analytical')
        res2, T0 = self.get_T_matrix(phi_ext_s_0=phi_ext_s_0,
                                     phi_ext_l_0=phi_ext_l_0, mode = 'analytical')
        w0, v0 = nl.eigh(T0)
        sqrtw = sl.sqrtm(np.diag(w0))
        # print(nl.norm(np.dot(np.dot(v0, np.diag(w0)), nl.inv(v0))-T0) \
        # /nl.norm(T0))
        U1 = np.dot(np.dot(nl.inv(v0), U0), v0)
        U2 = np.dot(np.dot(nl.inv(sqrtw), U1), nl.inv(sqrtw))
        w2, v2 = nl.eigh(U2)
        P = np.dot(np.dot(v0, nl.inv(sqrtw)), v2)
        invP = np.dot(np.dot(nl.inv(v2), nl.inv(sqrtw)), nl.inv(v0))
        U3 = np.dot(np.dot(invP, U0), P)
        return res1, res2, P, w2
