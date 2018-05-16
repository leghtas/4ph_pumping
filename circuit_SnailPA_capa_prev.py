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



def restrict_m(A, loc):
    dim = len(np.shape(A))
    _A = A
    indices = []
    for ii, elt in enumerate(loc):
        if elt == 0:
            indices.append(ii)

    for ii, index in enumerate(indices):
        index -= ii
        for jj in range(dim):
            _A = np.delete(_A, index, jj)
    return(_A)

def restrict_p(A, loc):
    dim = len(np.shape(A))
    _A = A
    indices = []
    for ii, elt in enumerate(loc):
        if elt == 0:
            indices.append(ii)

    for ii, index in enumerate(indices):
        index -= ii
        _A = np.delete(_A, index, 1)
    return(_A)


class CircuitSnailPA(Circuit):

    def __init__(self, EC, EL, EJ, alpha, n,
                 printParams=True):

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

    def get_U(self, phi_ext_0=0):
        def U(p, P=np.identity(2)):
            p = P.dot(p)
            (ps, pr) = (p[0], p[1]) #phi_snail and phi_resonator
            _U = (self.EL/hbar)*pr**2 + \
                 -self.alpha*(self.EJ/hbar)*np.cos(ps) +\
                 -self.n*(self.EJ/hbar)*np.cos((phi_ext_0-ps)/self.n)
            return _U
        return U

    def get_U_1d(self, phi_ext_0=0):
        def U(p, P=np.identity(2)):
            p = P.dot(p)
            (ps, pr) = (p[0], p[1]) #phi_snail and phi_resonator
            _U = -self.alpha*(self.EJ/hbar)*np.cos(ps) +\
                 -self.n*(self.EJ/hbar)*np.cos((phi_ext_0-ps)/self.n)
            return _U
        return U

    def get_dUs(self, phi_ext_0=0):
        def dUs(p, P=np.identity(2)):
            p = P.dot(p)
            (ps, pr) = (p[0], p[1])
            _dUs = self.alpha*(self.EJ/hbar)*np.sin(ps) +\
                 -(self.EJ/hbar)*np.sin((phi_ext_0-ps)/self.n)
            return _dUs
        return dUs

    def get_dUr(self, phi_ext_0=0):
        def dUr(p, P=np.identity(2)):
            p = P.dot(p)
            (ps, pr) = (p[0], p[1])
            _dUr = 2*(self.EL/hbar)*pr
            return _dUr
        return dUr

    def get_d2Uss(self, phi_ext_0=0):
        def d2Uss(p, P=np.identity(2)):
            p = P.dot(p)
            (ps, pr) = (p[0], p[1])
            _d2Uss = self.alpha*(self.EJ/hbar)*np.cos(ps) +\
                 1/self.n*(self.EJ/hbar)*np.cos((phi_ext_0-ps)/self.n)
            return _d2Uss
        return d2Uss

    def get_d2Urr(self, phi_ext_0=0):
        def d2Urr(p, P=np.identity(2)):
            p = P.dot(p)
            (ps, pr) = (p[0], p[1])
            _d2Urr = 2*(self.EL/hbar)
            return _d2Urr
        return d2Urr

    def get_HessU(self, phi_ext_0=0):
        def HessU(p, P=np.identity(2)):
#            p = P.dot(p)
            (ps, pr) = (p[0], p[1])
            d2Urr_p = self.get_d2Urr(phi_ext_0=phi_ext_0)([ps, pr])
            d2Uss_p = self.get_d2Uss(phi_ext_0=phi_ext_0)([ps, pr])
            _HessU = np.array([[d2Uss_p, 0],
                               [0, d2Urr_p]])
            _HessU1 = np.transpose(np.dot(np.transpose(_HessU, (0, 1)), P), (0, 1))
            _HessU2 = np.transpose(np.dot(np.transpose(_HessU1, (1, 0)), P), (1, 0))
            _HessU_basis = np.dot(np.dot(P.T, _HessU), P)
            _HessU_basis = _HessU2
            return _HessU_basis
        return HessU

    def get_d3U(self, phi_ext_0=0):
        def d3U(p, P=np.identity(2)):
            p = P.dot(p)
            (ps, pr) = (p[0], p[1])
            _d3U = -self.alpha*(self.EJ/hbar)*np.sin(ps) +\
                   1/self.n**2*(self.EJ/hbar)*np.sin((phi_ext_0-ps)/self.n)
            return _d3U
        return d3U

    def get_Hess3U(self, phi_ext_0=0):
        def Hess3U(p, P=np.identity(2)):
#            p = P.dot(p)
            (ps, pr) = (p[0], p[1])
            d3U_p = self.get_d3U(phi_ext_0=phi_ext_0)([ps, pr])
            _Hess3U = np.array([[[d3U_p, 0],
                                 [0, 0]],
                                [[0, 0],
                                 [0, 0]]])
            _Hess3U1 = np.transpose(np.dot(np.transpose(_Hess3U,(0,1,2)), P),(0,1,2))
            _Hess3U2 = np.transpose(np.dot(np.transpose(_Hess3U1,(0,2,1)), P),(0,2,1))
            _Hess3U3 = np.transpose(np.dot(np.transpose(_Hess3U2,(2,1,0)), P),(2,1,0))
            return _Hess3U3
        return Hess3U

    def get_d4U(self, phi_ext_0=0):
        def d4U(p, P=np.identity(2)):
            p = P.dot(p)
            (ps, pr) = (p[0], p[1])
            _d4U = -self.alpha*(self.EJ/hbar)*np.cos(ps) +\
                 -1/self.n**3*(self.EJ/hbar)*np.cos((phi_ext_0-ps)/self.n)
            return _d4U
        return d4U

    def get_Hess4U(self, phi_ext_0=0):
        def Hess4U(p, P=np.identity(2)):
#            p = P.dot(p)
            (ps, pr) = (p[0], p[1])
            d4U_p = self.get_d4U(phi_ext_0=phi_ext_0)([ps, pr])
            _Hess4U = np.array([[[[d4U_p, 0],
                                  [0, 0]],
                                 [[0, 0],
                                  [0, 0]]],
                                [[[0, 0],
                                  [0, 0]],
                                 [[0, 0],
                                  [0, 0]]]])
            _Hess4U1 = np.transpose(np.dot(np.transpose(_Hess4U,(0,1,2,3)), P),(0,1,2,3))
            _Hess4U2 = np.transpose(np.dot(np.transpose(_Hess4U1,(0,1,3,2)), P),(0,1,3,2))
            _Hess4U3 = np.transpose(np.dot(np.transpose(_Hess4U2,(0,3,2,1)), P),(0,3,2,1))
            _Hess4U4 = np.transpose(np.dot(np.transpose(_Hess4U3,(3,1,2,0)), P),(3,1,2,0))
            return _Hess4U4
        return Hess4U

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

    def get_U_matrix(self, phi_ext_0=0, mode = 'analytical'):
        U = self.get_U(phi_ext_0=phi_ext_0)
#        print(U([2,1]))
        if mode == 'analytical':
            x0 = np.array([0, 0])
            def U1(x):
                return U(x)/1e14
            res = minimize(U1, x0, method='SLSQP', tol=1e-12, bounds=[(-3*np.pi, 3*np.pi), (-3*np.pi, 3*np.pi)]) ################################################################# becareful bounds
            if not res.success:
                warnings.warn('Did not minimized', UserWarning)
            HessU = self.get_HessU(phi_ext_0=phi_ext_0)
            quad = res.x, HessU([res.x[0], res.x[1]])/2
#            print(res.x)
#            print(quad)
        else:
            quad = self.get_quadratic_form(U)
        return quad

    def get_T_matrix(self, phi_ext_0=0, mode = 'analytical'):
        T = self.get_T(phi_ext_0=phi_ext_0)
        print(T([2,1]))
        if mode == 'analytical':
            res = np.array([0, 0])
            HessT = self.get_HessT(phi_ext_0=phi_ext_0)
            quad = res, HessT([res[0], res[1]])/2
        else:
            quad = self.get_quadratic_form(T)
        return quad

    def get_freqs_kerrs(self, phi_ext_0=0):
        res = self.get_normal_mode_frame(phi_ext_0=phi_ext_0)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi

        # calculate Kerrs from polynomial approximation of potential
        U = self.get_U(phi_ext_0=phi_ext_0)


        HessU = self.get_HessU(phi_ext_0=phi_ext_0)
        Hess3U = self.get_Hess3U(phi_ext_0=phi_ext_0)
        Hess4U = self.get_Hess4U(phi_ext_0=phi_ext_0)

        Hess_r = HessU([res1[0], res1[1]], P=P)
        Hess3_r = Hess3U([res1[0], res1[1]], P=P)
        Hess4_r = Hess4U([res1[0], res1[1]], P=P)

        popt2 = np.array([Hess_r[0, 0]/2, Hess_r[1, 1]/2])
        popt3 = np.array([Hess3_r[0, 0, 0]/6, Hess3_r[1, 1, 1]/6]) # coeff devant le phi**3
        popt4 = np.array([Hess4_r[0, 0, 0, 0]/24, Hess4_r[1, 1, 1, 1]/24]) # coeff devant le phi**4

#        # In case we have too many degrees of freedom some modes do not exist
#        # ZPF then becomes infinite so we set it to 0
#        ZPF = np.zeros(2)
#        for ii in range(len(popt2)):
#            if popt2[ii]>0:
#                ZPF[ii] = popt2[ii]**(-1./4)
#            else:
#                ZPF[ii] = 0

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

    def get_freqs_only(self, phi_ext_0=0):
        res = self.get_normal_mode_frame(phi_ext_0=phi_ext_0)
        res1, res2, P, w2 = res
        fs = np.sqrt(w2)/2/np.pi
        return fs

    def get_normal_mode_frame(self, phi_ext_0=0):

        res1, U0 = self.get_U_matrix(phi_ext_0=phi_ext_0, mode = 'analytical')
        res2, T0 = self.get_T_matrix(phi_ext_0=phi_ext_0, mode = 'analytical')
        
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
        
#        print(res1, U0)
#        print(res2, T0)
        return res1, res2, P, wU3

def true0(A):
    A_max = np.max(np.abs(A))
#    print(A_max)
    shape = np.shape(A)
    n_elts =1
    for ii in shape:
        n_elts *= ii
#    print(n_elts)
    A_r = A.reshape(n_elts)

    for ii, elt  in enumerate(A_r):
        if np.abs(elt/A_max) <1e-12:
            A_r[ii] = 0
    _A = A_r.reshape(shape)
    return _A

#_Hess3U = np.array([[1, 0],
#                    [0, 0]])
#_Hess3U = np.array([[[1, 0],
#                     [0, 0]],
#                    [[0, 0],
#                     [0, 0]]])
#P = 1/2**0.5*np.array([[ 1, -1],
#                       [ 1,  1]])
#
#_HessU1 = np.transpose(np.dot(np.transpose(_HessU, (0, 1)), P), (0, 1))
#_HessU2 = np.transpose(np.dot(np.transpose(_HessU1, (1, 0)), P), (1, 0))
#
#_Hess3U1 = np.transpose(np.dot(np.transpose(_Hess3U,(0,1,2)), P),(0,1,2))
#_Hess3U2 = np.transpose(np.dot(np.transpose(_Hess3U1,(0,2,1)), P),(0,2,1))
#_Hess3U3 = np.transpose(np.dot(np.transpose(_Hess3U2,(2,1,0)), P),(2,1,0))
#
#print(_HessU1)
#print(_HessU2)
#
#print(_Hess3U1)
#print(_Hess3U2)
#print(_Hess3U3)
#
#print('test')
#print(np.dot(_Hess3U, P))
#print(np.transpose(np.dot(_Hess3U, P), (1,0,2)))

print(hbar)
