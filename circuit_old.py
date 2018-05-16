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


class Circuit(object):

    def __init__(self, ECa, ELa, ECb, ELb, EJ1, EJ2, na, nb, Eg=None,
                 ECc=None, ELc=None, Ecoup=None, nc=None,
                 printParams=True):
        self.na = na
        self.nb = nb
        self.nc = nc
        self.ida = qt.identity(na)
        self.idb = qt.identity(nb)
        a = qt.destroy(na)
        b = qt.destroy(nb)
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
        self.Eg = Eg
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
        self.Phia_1 = phia*(a+a.dag())
        self.Phib_1 = phib*(b+b.dag())
        self.Na_1 = na_zpf*(a-a.dag())/1j
        self.Nb_1 = nb_zpf*(b-b.dag())/1j

        if printParams:
            print("La = "+str(La*1e9)+" nH")
            print("Ca = "+str(Ca*1e15)+" fF")
            print("Lb = "+str(Lb*1e9)+" nH")
            print("Cb = "+str(Cb*1e15)+" fF")
            print("ELa/h = "+str(1e-9*self.ELa/hbar/2/pi)+" GHz")
            print("ELb/h = "+str(1e-9*self.ELb/hbar/2/pi)+" GHz")
            print("ELc/h = "+str(1e-9*self.ELc/hbar/2/pi)+" GHz")
            print("ECa/h = "+str(1e-9*self.ECa/hbar/2/pi)+" GHz")
            print("ECb/h = "+str(1e-9*self.ECb/hbar/2/pi)+" GHz")
            print("ECc/h = "+str(1e-9*self.ECc/hbar/2/pi)+" GHz")
            print("EJ/h = "+str(1e-9*self.EJ/hbar/2/pi)+" GHz")
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

    def getH_overhbar_cpb_cav(self):
        ECa = self.ECa
        ECb = self.ECb
        ELa = self.ELa
        Eg = self.Eg
        Na_1 = qt.tensor(self.Na_1, self.idb)
        Nb_1 = qt.tensor(self.ida, self.Nb_1)
        Phia_1 = qt.tensor(self.Phia_1, self.idb)
        Phib_1 = qt.tensor(self.ida, self.Phib_1)
        EJ = self.EJ

        H0a = 4*(ECa/hbar)*(Na_1)**2+0.5*(ELa/hbar)*Phia_1**2
        H0b = 4*(ECb/hbar)*(Nb_1)**2
        Hg = Eg*Na_1*Nb_1
        H0 = H0a + H0b + Hg
        H1 = -(EJ/hbar) * cosm(Phib_1)
        H2 = +(EJ/hbar) * sinm(Phib_1)

        return H0, H1, H2

    def getH_overhbar_two_mode(self, phi_ext_s=0, phi_ext_l=0):
        ECa = self.ECa
        ECb = self.ECb
        ELa = self.ELa
        ELb = self.ELb
        Na_1 = qt.tensor(self.Na_1, self.idb)
        Nb_1 = qt.tensor(self.ida, self.Nb_1)
        Phia_1 = qt.tensor(self.Phia_1, self.idb)
        Phib_1 = qt.tensor(self.ida, self.Phib_1)
        EJ = self.EJ

        Phiext_s = phi_ext_s*qt.tensor(self.ida, self.idb)
        Phiext_l = phi_ext_l*qt.tensor(self.ida, self.idb)

        H0a = 4*(ECa/hbar)*(Na_1)**2+0.5*(ELa/hbar)*Phia_1**2
        H0b = 4*(ECb/hbar)*(Nb_1)**2+0.5*(ELb/hbar)*Phib_1**2
        Hc = (-(EJ/hbar)*cosm(Phiext_s/2) *  # each junction is 2*LJ
              cosm(Phia_1+Phib_1+Phiext_s/2+Phiext_l))
        H_over_hbar_two_modes = (H0a + H0b + Hc)
        return H_over_hbar_two_modes

    def getH_overhbar_three_mode(self, phi_ext_s=0, phi_ext_l=0):
        if self.ECc is None or self.ELc is None or self.Ecoup is None:
            raise ValueError("Specify values for ECc, ELc and Ecoup")
        else:
            pass
        wc, Zc, LJ = get_w_Z_LJ_from_E(self.ECc, self.ELc, EJ=1)
        nc_zpf = (1/(2*e)) * (np.sqrt(hbar/2/Zc))  # nZPF
        phic = (1/phi0) * (np.sqrt((hbar/2)*Zc))  # phiZPF
        c = qt.destroy(self.nc)
        self.Nc_1 = nc_zpf*(c-c.dag())/1j
        self.Phic_1 = phic*(c+c.dag())
        self.idc = qt.identity(self.nc)

        ECa = self.ECa
        ECb = self.ECb
        ECc = self.ECc
        ELa = self.ELa
        ELb = self.ELb
        ELc = self.ELc
        Ecoup = self.Ecoup
        Na_1 = qt.tensor(self.Na_1, self.idb, self.idc)
        Nb_1 = qt.tensor(self.ida, self.Nb_1, self.idc)
        Nc_1 = qt.tensor(self.ida, self.idb, self.Nc_1)
        Phia_1 = qt.tensor(self.Phia_1, self.idb, self.idc)
        Phib_1 = qt.tensor(self.ida, self.Phib_1, self.idc)
        Phic_1 = qt.tensor(self.ida, self.idb, self.Phic_1)

        EJ = self.EJ

        Phiext_s = phi_ext_s*qt.tensor(self.ida, self.idb, self.idc)
        Phiext_l = phi_ext_l*qt.tensor(self.ida, self.idb, self.idc)

        H0a = 4*(ECa/hbar)*(Na_1)**2+0.5*(ELa/hbar)*Phia_1**2
        H0b = 4*(ECb/hbar)*(Nb_1)**2+0.5*(ELb/hbar)*Phib_1**2
        H0c = 4*(ECc/hbar)*(Nc_1)**2+0.5*(ELc/hbar)*Phic_1**2
        Hc = (-(EJ/hbar)*cosm(Phiext_s/2) *  # each junction is 2*LJ
              cosm(Phia_1+Phib_1+Phiext_s/2+Phiext_l)) \
            + 4*(Ecoup/hbar)*(Na_1 - Nc_1)**2
        H_over_hbar_three_modes = (H0a + H0b + H0c + Hc)
        return H_over_hbar_three_modes

    def getH_over_hbar_one_mode(self, phi_ext_s=0, phi_ext_l=0, n_g=0):
        na = self.na
        ECa = self.ECa
        ELa = self.ELa
        Na_1 = self.Na_1
        Ng = n_g*qt.qeye(self.na)
        Phia_1 = self.Phia_1
        EJ = self.EJ

        Phiext_s = phi_ext_s*qt.qeye(na)
        Phiext_l = phi_ext_l*qt.qeye(na)

        H0 = 4*(ECa/hbar)*(Na_1-Ng)**2+0.5*(ELa/hbar)*Phia_1**2
        Hc = (-(EJ/hbar)*cosm(Phiext_s/2) *  # each junction is 2*LJ
              cosm(Phia_1+Phiext_s/2+Phiext_l))
        H_over_hbar_one_mode = (H0 + Hc)
        return H_over_hbar_one_mode

    def get_U_over_hbar_one_mode(self, phiVec, phi_ext_s=0,
                                 phi_ext_l=0, n_g=0):
        ELa = self.ELa
        EJ = self.EJ
        U0 = np.empty(len(phiVec))
        U = np.empty(len(phiVec))
        for ii, phi in enumerate(phiVec):
            U[ii] = (0.5*(ELa/hbar)*phi**2 -
                     ((EJ/hbar)*np.cos(phi_ext_s/2) *
                      np.cos(phi+phi_ext_l+phi_ext_s/2)))
            U0[ii] = (0.5*(ELa/hbar)*phi**2 -
                      ((EJ/hbar)*np.cos(phi_ext_s/2) *
                       np.cos(phi_ext_l+phi_ext_s/2)*(1-phi**2/2)))
        return U0, U

    def get_U_over_hbar_two_modes(self, phiVec, phi_ext_s=0,
                                  phi_ext_l=0, n_g=0):
        ELa = self.ELa
        EJ = self.EJ
        U0 = np.empty(len(phiVec), len(phiVec))
        U = np.empty(len(phiVec), len(phiVec))
        for ii, phi in enumerate(phiVec):
            U[ii] = (0.5*(ELa/hbar)*phi**2 -
                     ((EJ/hbar)*np.cos(phi_ext_s/2) *
                      np.cos(phi+phi_ext_l+phi_ext_s/2)))
            U0[ii] = (0.5*(ELa/hbar)*phi**2 -
                      ((EJ/hbar)*np.cos(phi_ext_s/2) *
                       np.cos(phi_ext_l+phi_ext_s/2)*(1-phi**2/2)))
        return U0, U

    def getElevels_sweep_s(self, phi_ext_sweep,
                           phi_ext_l=0, ng=0):
        Elevels = np.empty((np.size(phi_ext_sweep), self.na))
        ws = np.empty(np.size(phi_ext_sweep))
        Kerrs = np.empty(np.size(phi_ext_sweep))
        minE = np.inf
        for kk in range(np.size(phi_ext_sweep)):
            phi_ext_s = phi_ext_sweep[kk]
            Hoverhbark = self.getH_over_hbar_one_mode(phi_ext_s=phi_ext_s,
                                                      phi_ext_l=phi_ext_l,
                                                      n_g=0)
            if minE > np.min(Hoverhbark.eigenenergies()):
                minE = np.min(Hoverhbark.eigenenergies())
            minEs = np.min(Hoverhbark.eigenenergies())*np.ones(self.na)
            Elevels[kk, :] = np.real(Hoverhbark.eigenenergies() - minEs)
            ws[kk] = np.diff(Elevels[kk, :])[0]
            Kerrs[kk] = np.diff(np.diff(Elevels[kk, :]))[0]
        Elevels = Elevels + minE
        return Elevels, ws, Kerrs

    def getElevels_sweep_l(self, phi_ext_sweep,
                           phi_ext_s=0, n_g=0):
        Elevels = np.zeros((np.size(phi_ext_sweep), self.na))
        ws = np.empty(np.size(phi_ext_sweep))
        Kerrs = np.empty(np.size(phi_ext_sweep))
        minE = np.inf
        for kk in range(np.size(phi_ext_sweep)):
            phi_ext_l = phi_ext_sweep[kk]
            Hoverhbark = self.getH_over_hbar_one_mode(phi_ext_s=phi_ext_s,
                                                      phi_ext_l=phi_ext_l,
                                                      n_g=0)
            if minE > np.min(Hoverhbark.eigenenergies()):
                minE = np.min(Hoverhbark.eigenenergies())
            minEs = np.min(Hoverhbark.eigenenergies())*np.ones(self.na)
            Elevels[kk, :] = np.real(Hoverhbark.eigenenergies() - minEs)
            ws[kk] = np.diff(Elevels[kk, :])[0]
            Kerrs[kk] = np.diff(np.diff(Elevels[kk, :]))[0]
        Elevels = Elevels + minE
        return Elevels, ws, Kerrs

    def getElevels_sweep_n(self, ng_sweep,
                           phi_ext_s=0, phi_ext_l=0):
        Elevels = np.zeros((np.size(ng_sweep), self.na))
        ws = np.empty(np.size(ng_sweep))
        Kerrs = np.empty(np.size(ng_sweep))
        minE = np.inf
        for kk in range(np.size(ng_sweep)):
            n_g = ng_sweep[kk]
            Hoverhbark = self.getH_over_hbar_one_mode(phi_ext_s=phi_ext_s,
                                                      phi_ext_l=phi_ext_l,
                                                      n_g=n_g)
            if minE > np.min(Hoverhbark.eigenenergies()):
                minE = np.min(Hoverhbark.eigenenergies())
            minEs = np.min(Hoverhbark.eigenenergies())*np.ones(self.na)
            Elevels[kk, :] = np.real(Hoverhbark.eigenenergies() - minEs)
            ws[kk] = np.diff(Elevels[kk, :])[0]
            Kerrs[kk] = np.diff(np.diff(Elevels[kk, :]))[0]
        Elevels = Elevels + minE
        return Elevels, ws, Kerrs

    def pltElevels_s(self, phi_ext_sweep, phiVec=np.linspace(-pi, pi, 101),
                     phi_ext_s_0=0, phi_ext_l_0=0):
        H = self.getH_over_hbar_one_mode(phi_ext_s=phi_ext_s_0,
                                         phi_ext_l=phi_ext_l_0)
        E = np.real(H.eigenenergies()/2/pi/1e9)
        E = E[:10]
        E = E*np.ones((len(phiVec), len(E)))
        U0, U = self.get_U_over_hbar_one_mode(phiVec, phi_ext_s=phi_ext_s_0,
                                              phi_ext_l=phi_ext_l_0)

        Elevels, ws, Kerrs = self.getElevels_sweep_s(phi_ext_sweep,
                                                     phi_ext_l=phi_ext_l_0)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
        ax[0].xaxis.set_tick_params(labelsize=ls)
        ax[1].xaxis.set_tick_params(labelsize=ls)
        ax[0].yaxis.set_tick_params(labelsize=ls)
        ax[1].yaxis.set_tick_params(labelsize=ls)
        ax[0].plot(phi_ext_sweep/2/pi, Elevels[:, :10]/(2*pi)/1e9)
        ax[0].set_xlabel('phiext_s/2pi', fontsize=fs)
        ax[0].set_ylabel('Energy levels (GHz)', fontsize=fs)
        ax[0].set_title('phi_ext_l/2pi = '+str(phi_ext_l_0/2/pi), fontsize=fs)
        ax[1].plot(phiVec/2/pi, U/2/pi/1e9)
        ax[1].plot(phiVec/2/pi, U0/2/pi/1e9)
        ax[1].plot(phiVec/2/pi, E)
        ax[1].set_ylim([np.min(U/2/pi/1e9), np.max(E)])
        ax[1].set_xlabel('phi/2pi', fontsize=fs)
        ax[1].set_ylabel('Potential energy (h x GHz)', fontsize=fs)
        title = ('Potential energy at \n' +
                 'phiext_s/2pi = '+str(phi_ext_s_0/2/pi) +
                 ', phiext_l/2pi = '+str(phi_ext_l_0/2/pi))
        ax[1].set_title(title, fontsize=fs)
        plt.savefig(('figures/'+'Elevels_vs_phiexts_at_phiextl_over2pi=' +
                     str(phi_ext_l_0/2/pi).replace('.', '_')+'.pdf'))
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].xaxis.set_tick_params(labelsize=ls)
        ax[1].xaxis.set_tick_params(labelsize=ls)
        ax[0].yaxis.set_tick_params(labelsize=ls)
        ax[1].yaxis.set_tick_params(labelsize=ls)
        ax[0].plot(phi_ext_sweep/2/pi, ws/2/pi/1e9)
        ax[0].set_ylabel('Frequency (GHz)', fontsize=fs)
        ax[1].plot(phi_ext_sweep/2/pi, Kerrs/2/pi/1e6)
        ax[1].set_ylabel('Kerr (MHz)', fontsize=fs)
        ax[1].set_xlabel('phiext_s/2pi', fontsize=fs)
        title = 'phiext_l/2pi = '+str(phi_ext_l_0/2/pi)
        fig.suptitle(title, fontsize=fs)
        plt.savefig(('figures/'+'Freq_Kerr_vs_phiexts_at_phiextl_over2pi=' +
                     str(phi_ext_l_0/2/pi).replace('.', '_')+'.pdf'))

    def pltElevels_l(self, phi_ext_sweep, phiVec=np.linspace(-pi, pi, 101),
                     phi_ext_s_0=0, phi_ext_l_0=0):
        H = self.getH_over_hbar_one_mode(phi_ext_s=phi_ext_s_0,
                                         phi_ext_l=phi_ext_l_0)
        E = np.real(H.eigenenergies()/2/pi/1e9)
        E = E[:10]
        E = E*np.ones((len(phiVec), len(E)))
        U0, U = self.get_U_over_hbar_one_mode(phiVec, phi_ext_s=phi_ext_s_0,
                                              phi_ext_l=phi_ext_l_0)

        Elevels, ws, Kerrs = self.getElevels_sweep_l(phi_ext_sweep,
                                                     phi_ext_s=phi_ext_s_0)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
        ax[0].xaxis.set_tick_params(labelsize=ls)
        ax[1].xaxis.set_tick_params(labelsize=ls)
        ax[0].yaxis.set_tick_params(labelsize=ls)
        ax[1].yaxis.set_tick_params(labelsize=ls)
        ax[0].plot(phi_ext_sweep/2/pi, Elevels[:, :10]/(2*pi)/1e9)
        ax[0].set_xlabel('phiext_l/2pi', fontsize=fs)
        ax[0].set_ylabel('Energy levels (GHz)', fontsize=fs)
        ax[0].set_title('phi_ext_s/2pi = '+str(phi_ext_s_0/2/pi), fontsize=fs)
        ax[1].plot(phiVec/2/pi, U/2/pi/1e9)
        ax[1].plot(phiVec/2/pi, U0/2/pi/1e9)
        ax[1].plot(phiVec/2/pi, E)
        ax[1].set_ylim([np.min(U/2/pi/1e9), np.max(E)])
        ax[1].set_xlabel('phi/2pi', fontsize=fs)
        ax[1].set_ylabel('Potential energy (h x GHz)', fontsize=fs)
        title = ('Potential energy at \n' +
                 'phiext_s/2pi = '+str(phi_ext_s_0/2/pi) +
                 ', phiext_l/2pi = '+str(phi_ext_l_0/2/pi))
        ax[1].set_title(title, fontsize=fs)
        plt.savefig(('figures/'+'Elevels_vs_phiextl_at_phiexts_over2pi=' +
                     str(phi_ext_s_0/2/pi).replace('.', '_')+'.pdf'))

        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
        ax[0].xaxis.set_tick_params(labelsize=ls)
        ax[1].xaxis.set_tick_params(labelsize=ls)
        ax[0].yaxis.set_tick_params(labelsize=ls)
        ax[1].yaxis.set_tick_params(labelsize=ls)
        ax[0].plot(phi_ext_sweep/2/pi, ws/2/pi/1e9)
        ax[0].set_ylabel('Frequency (GHz)', fontsize=fs)
        ax[1].plot(phi_ext_sweep/2/pi, Kerrs/2/pi/1e6)
        ax[1].set_ylabel('Kerr (MHz)', fontsize=fs)
        ax[1].set_xlabel('phiext_l/2pi', fontsize=fs)
        title = 'phiext_s/2pi = '+str(phi_ext_s_0/2/pi)
        fig.suptitle(title, fontsize=fs)
        plt.savefig(('figures/'+'Freq_Kerr_vs_phiextl_at_phiexts_over2pi=' +
                     str(phi_ext_s_0/2/pi).replace('.', '_')+'.pdf'))

    def pltElevels_n(self, ng_sweep, phiVec=np.linspace(-pi, pi, 101),
                     phi_ext_s_0=0, phi_ext_l_0=0, n_g=0):
        H = self.getH_over_hbar_one_mode(phi_ext_s=phi_ext_s_0,
                                         phi_ext_l=phi_ext_l_0)
        E = np.real(H.eigenenergies()/2/pi/1e9)
        E = E[:10]
        E = E*np.ones((len(phiVec), len(E)))
        U0, U = self.get_U_over_hbar_one_mode(phiVec, phi_ext_s=phi_ext_s_0,
                                              phi_ext_l=phi_ext_l_0)

        Elevels, ws, Kerrs = self.getElevels_sweep_n(ng_sweep,
                                                     phi_ext_s=phi_ext_s_0,
                                                     phi_ext_l=phi_ext_l_0)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
        ax[0].xaxis.set_tick_params(labelsize=ls)
        ax[1].xaxis.set_tick_params(labelsize=ls)
        ax[0].yaxis.set_tick_params(labelsize=ls)
        ax[1].yaxis.set_tick_params(labelsize=ls)
        ax[0].plot(ng_sweep, Elevels[:, :10]/(2*pi)/1e9)
        ax[0].set_xlabel('n_gate', fontsize=fs)
        ax[0].set_ylabel('Energy levels (GHz)', fontsize=fs)
        ax[0].set_title('phi_ext_s/2pi = '+str(phi_ext_s_0/2/pi)+'\n' +
                        'phi_ext_l/2pi = '+str(phi_ext_l_0/2/pi), fontsize=fs)
        ax[1].plot(phiVec/2/pi, U/2/pi/1e9)
        ax[1].plot(phiVec/2/pi, U0/2/pi/1e9)
        ax[1].plot(phiVec/2/pi, E)
        ax[1].set_ylim([np.min(U/2/pi/1e9), np.max(E)])
        ax[1].set_xlabel('phi/2pi', fontsize=fs)
        ax[1].set_ylabel('Potential energy (h x GHz)', fontsize=fs)
        title = ('Potential energy at \n' +
                 'phiext_s/2pi = '+str(phi_ext_s_0/2/pi) +
                 ', phiext_l/2pi = '+str(phi_ext_l_0/2/pi))
        ax[1].set_title(title, fontsize=fs)
        plt.savefig(('figures/'+'Elevels_vs_ng_at_' +
                     'phiexts_over2pi=' +
                     str(phi_ext_s_0/2/pi).replace('.', '_')+'_'
                     'phiextl_over2pi=' +
                     str(phi_ext_l_0/2/pi).replace('.', '_')+'.pdf'))
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].xaxis.set_tick_params(labelsize=ls)
        ax[1].xaxis.set_tick_params(labelsize=ls)
        ax[0].yaxis.set_tick_params(labelsize=ls)
        ax[1].yaxis.set_tick_params(labelsize=ls)
        ax[0].plot(ng_sweep/2/pi, ws/2/pi/1e9)
        ax[0].set_ylabel('Frequency (GHz)', fontsize=fs)
        ax[1].plot(ng_sweep/2/pi, Kerrs/2/pi/1e6)
        ax[1].set_ylabel('Kerr (MHz)', fontsize=fs)
        ax[1].set_xlabel('n_gate', fontsize=fs)
        title = 'phiext_s/2pi = '+str(phi_ext_s_0/2/pi)
        fig.suptitle(title, fontsize=fs)
        plt.savefig(('figures/'+'Freq_Kerr_vs_ng_at_' +
                     'phiexts_over2pi=' +
                     str(phi_ext_s_0/2/pi).replace('.', '_')+'_'
                     'phiextl_over2pi=' +
                     str(phi_ext_l_0/2/pi).replace('.', '_')+'.pdf'))

    def pltElevels(self, sweep_type, sweep_values,
                   phiVec=np.linspace(-pi, pi, 101),
                   phi_ext_s_0=0, phi_ext_l_0=0, n_g_0=0):

        if sweep_type not in ['phi_ext_s', 'phi_ext_l', 'n_g']:
            raise Exception
        H = self.getH_over_hbar_one_mode(phi_ext_s=phi_ext_s_0,
                                         phi_ext_l=phi_ext_l_0,
                                         n_g=n_g_0)
        E = np.real(H.eigenenergies()/2/pi/1e9)
        E = E[:10]
        E = E*np.ones((len(phiVec), len(E)))
        U0, U = self.get_U_over_hbar_one_mode(phiVec, phi_ext_s=phi_ext_s_0,
                                              phi_ext_l=phi_ext_l_0, n_g=n_g_0)

        if sweep_type == 'phi_ext_s':
            Elevels, ws, Kerrs = self.getElevels_sweep_s(sweep_values,
                                                         phi_ext_l=phi_ext_l_0)
            xlabel = 'phiext_s/2pi'
            xvec = sweep_values/2/pi
            title1 = 'phi_ext_l/2pi = '+str(phi_ext_l_0/2/pi)

        if sweep_type == 'phi_ext_l':
            Elevels, ws, Kerrs = self.getElevels_sweep_l(sweep_values,
                                                         phi_ext_s=phi_ext_s_0)
            xlabel = 'phiext_l/2pi'
            xvec = sweep_values/2/pi
            title1 = 'phi_ext_s/2pi = '+str(phi_ext_s_0/2/pi)

        if sweep_type == 'n_g':
            Elevels, ws, Kerrs = self.getElevels_sweep_n(sweep_values,
                                                         phi_ext_s=phi_ext_s_0,
                                                         phi_ext_l=phi_ext_l_0)
            xlabel = 'n_gate'
            xvec = sweep_values
            title1 = 'phi_ext_s/2pi = '+str(phi_ext_s_0/2/pi) + \
                     ', phi_ext_l/2pi = '+str(phi_ext_l_0/2/pi)

        title2 = ((title1.replace('.', '_')).replace('/', 'over')).replace(' ',
                                                                           '')

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.xaxis.set_tick_params(labelsize=ls)
        ax.yaxis.set_tick_params(labelsize=ls)
        ax.plot(xvec, Elevels[:, :10]/(2*pi)/1e9)
        ax.set_xlabel(xlabel, fontsize=fs)
        ax.set_ylabel('Transition frequencies (GHz)', fontsize=fs)
        ax.set_title(title1, fontsize=fs)
        plt.savefig(('figures/'+'Elevels_vs_'+sweep_type +
                     '_at_'+title2+'.pdf'))

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.xaxis.set_tick_params(labelsize=ls)
        ax.yaxis.set_tick_params(labelsize=ls)
        ax.plot(phiVec/2/pi, U/2/pi/1e9)
        ax.plot(phiVec/2/pi, U0/2/pi/1e9)
        ax.plot(phiVec/2/pi, E)
        ax.set_ylim([np.min(U/2/pi/1e9), np.max(E)])
        ax.set_xlabel('phi/2pi', fontsize=fs)
        ax.set_ylabel('Potential energy (h x GHz)', fontsize=fs)
        title = ('Potential energy at \n' +
                 'phiext_s/2pi = '+str(phi_ext_s_0/2/pi) +
                 ', phiext_l/2pi = '+str(phi_ext_l_0/2/pi))
        ax.set_title(title, fontsize=fs)
        plt.savefig(('figures/'+'Potential' +
                     '_at_'+title2+'.pdf'))

        fig, ax = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        ax[0].xaxis.set_tick_params(labelsize=ls)
        ax[1].xaxis.set_tick_params(labelsize=ls)
        ax[0].yaxis.set_tick_params(labelsize=ls)
        ax[1].yaxis.set_tick_params(labelsize=ls)
        ax[0].plot(xvec, ws/2/pi/1e9)
        ax[0].set_ylabel('Frequency (GHz)', fontsize=fs)
        ax[1].plot(xvec, Kerrs/2/pi/1e6)
        ax[1].set_ylabel('Kerr (MHz)', fontsize=fs)
        ax[1].set_xlabel(xlabel, fontsize=fs)
        fig.suptitle(title1, fontsize=fs)
        plt.savefig(('figures/'+'Freq_Kerr_vs_' +
                     sweep_type+'_at_'+title2+'.pdf'))
        return Elevels

    def get_U(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def U(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            _U = 0.5*(self.ELa/hbar)*pa**2 + \
                 0.5*(self.ELb/hbar)*pb**2 + \
                 0.5*(self.ELc/hbar)*pc**2 + \
                 -(self.EJ/hbar)*np.cos(phi_ext_s_0/2) * \
                 np.cos(pa+pb+phi_ext_s_0/2+phi_ext_l_0) + \
                 -(self.dEJ/hbar)*np.sin(phi_ext_s_0/2) * \
                 np.sin(pa+pb+phi_ext_s_0/2+phi_ext_l_0)
            return _U
        return U

    def get_dUa(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def dUa(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            _dUa = (self.ELa/hbar)*pa + \
                 (self.EJ/hbar)*np.cos(phi_ext_s_0/2) * \
                 np.sin(pa+pb+phi_ext_s_0/2+phi_ext_l_0) + \
                 -(self.dEJ/hbar)*np.sin(phi_ext_s_0/2) * \
                 np.cos(pa+pb+phi_ext_s_0/2+phi_ext_l_0)
            return _dUa
        return dUa

    def get_dUb(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def dUb(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            _dUb = (self.ELb/hbar)*pb + \
                 (self.EJ/hbar)*np.cos(phi_ext_s_0/2) * \
                 np.sin(pa+pb+phi_ext_s_0/2+phi_ext_l_0) + \
                 -(self.dEJ/hbar)*np.sin(phi_ext_s_0/2) * \
                 np.cos(pa+pb+phi_ext_s_0/2+phi_ext_l_0)
            return _dUb
        return dUb

    def get_dUc(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def dUc(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            _dUc = (self.ELc/hbar)*pc
            return _dUc
        return dUc

    def get_d2Uaa(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def d2Uaa(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            _d2Uaa = (self.ELa/hbar) + \
                 (self.EJ/hbar)*np.cos(phi_ext_s_0/2) * \
                 np.cos(pa+pb+phi_ext_s_0/2+phi_ext_l_0) + \
                 (self.dEJ/hbar)*np.sin(phi_ext_s_0/2) * \
                 np.sin(pa+pb+phi_ext_s_0/2+phi_ext_l_0)
            return _d2Uaa
        return d2Uaa

    def get_d2Ubb(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def d2Ubb(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            _d2Ubb = (self.ELb/hbar) + \
                 (self.EJ/hbar)*np.cos(phi_ext_s_0/2) * \
                 np.cos(pa+pb+phi_ext_s_0/2+phi_ext_l_0) + \
                 (self.dEJ/hbar)*np.sin(phi_ext_s_0/2) * \
                 np.sin(pa+pb+phi_ext_s_0/2+phi_ext_l_0)
            return _d2Ubb
        return d2Ubb

    def get_d2Ucc(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def d2Ucc(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            _d2Ucc = (self.ELc/hbar)
            return _d2Ucc
        return d2Ucc

    def get_d2Uab(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def d2Uab(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            _d2Uab = (self.EJ/hbar)*np.cos(phi_ext_s_0/2) * \
                 np.cos(pa+pb+phi_ext_s_0/2+phi_ext_l_0) + \
                 (self.dEJ/hbar)*np.sin(phi_ext_s_0/2) * \
                 np.sin(pa+pb+phi_ext_s_0/2+phi_ext_l_0)
            return _d2Uab
        return d2Uab

    def get_d2Uac(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def d2Uac(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            _d2Uac = 0
            return _d2Uac
        return d2Uac

    def get_d2Ubc(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def d2Ubc(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            _d2Ubc = 0
            return _d2Ubc
        return d2Ubc

    def get_HessU(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def HessU(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            d2Uaa_p = self.get_d2Uaa(phi_ext_s_0=phi_ext_s_0,
                                   phi_ext_l_0=phi_ext_l_0)([pa, pb, pc])
            d2Ubb_p = self.get_d2Ubb(phi_ext_s_0=phi_ext_s_0,
                                   phi_ext_l_0=phi_ext_l_0)([pa, pb, pc])
            d2Ucc_p = self.get_d2Ucc(phi_ext_s_0=phi_ext_s_0,
                                   phi_ext_l_0=phi_ext_l_0)([pa, pb, pc])
            d2Uab_p = self.get_d2Uab(phi_ext_s_0=phi_ext_s_0,
                                   phi_ext_l_0=phi_ext_l_0)([pa, pb, pc])
            d2Uac_p = self.get_d2Uac(phi_ext_s_0=phi_ext_s_0,
                                   phi_ext_l_0=phi_ext_l_0)([pa, pb, pc])
            d2Ubc_p = self.get_d2Ubc(phi_ext_s_0=phi_ext_s_0,
                                   phi_ext_l_0=phi_ext_l_0)([pa, pb, pc])
            _HessU = np.array([[d2Uaa_p, d2Uab_p, d2Uac_p],
                               [d2Uab_p, d2Ubb_p, d2Ubc_p],
                               [d2Uac_p, d2Ubc_p, d2Ucc_p]])
            _HessU1 = np.transpose(np.dot(np.transpose(_HessU,(0,1)), P),(0,1))
            _HessU2 = np.transpose(np.dot(np.transpose(_HessU1,(1,0)), P),(1,0))
            _HessU_basis = np.dot(np.dot(P.T, _HessU), P)
            _HessU_basis = _HessU2
            return _HessU_basis
        return HessU

    def get_d3U(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def d3U(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            _d3U = -(self.EJ/hbar)*np.cos(phi_ext_s_0/2) * \
                 np.sin(pa+pb+phi_ext_s_0/2+phi_ext_l_0) + \
                 (self.dEJ/hbar)*np.sin(phi_ext_s_0/2) * \
                 np.cos(pa+pb+phi_ext_s_0/2+phi_ext_l_0)
            return _d3U
        return d3U

    def get_Hess3U(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def Hess3U(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            d3U_p = self.get_d3U(phi_ext_s_0=phi_ext_s_0,
                                   phi_ext_l_0=phi_ext_l_0)([pa, pb, pc])
            _Hess3U = np.array([[[d3U_p, d3U_p, 0],
                                 [d3U_p, d3U_p, 0],
                                 [0, 0, 0]],
                                [[d3U_p, d3U_p, 0],
                                 [d3U_p, d3U_p, 0],
                                 [0, 0, 0]],
                                [[0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0]]])
            _Hess3U1 = np.transpose(np.dot(np.transpose(_Hess3U,(0,1,2)), P),(0,1,2))
            _Hess3U2 = np.transpose(np.dot(np.transpose(_Hess3U1,(1,0,2)), P),(1,0,2))
            _Hess3U3 = np.transpose(np.dot(np.transpose(_Hess3U2,(2,1,0)), P),(2,1,0))
            return _Hess3U3
        return Hess3U

    def get_d4U(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def d4U(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            _d4U = -(self.EJ/hbar)*np.cos(phi_ext_s_0/2) * \
                 np.cos(pa+pb+phi_ext_s_0/2+phi_ext_l_0) + \
                 -(self.dEJ/hbar)*np.sin(phi_ext_s_0/2) * \
                 np.sin(pa+pb+phi_ext_s_0/2+phi_ext_l_0)
            return _d4U
        return d4U

    def get_Hess4U(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def Hess4U(p, P=np.identity(3)):
            p = P.dot(p)
            (pa, pb, pc) = (p[0], p[1], p[2])
            d4U_p = self.get_d4U(phi_ext_s_0=phi_ext_s_0,
                                   phi_ext_l_0=phi_ext_l_0)([pa, pb, pc])
            _Hess4U = np.array([[[[d4U_p, d4U_p, 0],
                                  [d4U_p, d4U_p, 0],
                                  [0, 0, 0]],
                                 [[d4U_p, d4U_p, 0],
                                  [d4U_p, d4U_p, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]]],
                                [[[d4U_p, d4U_p, 0],
                                  [d4U_p, d4U_p, 0],
                                  [0, 0, 0]],
                                 [[d4U_p, d4U_p, 0],
                                  [d4U_p, d4U_p, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]]],
                                [[[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]]]])
            _Hess4U1 = np.transpose(np.dot(np.transpose(_Hess4U,(0,1,2,3)), P),(0,1,2,3))
            _Hess4U2 = np.transpose(np.dot(np.transpose(_Hess4U1,(1,0,2,3)), P),(1,0,2,3))
            _Hess4U3 = np.transpose(np.dot(np.transpose(_Hess4U2,(2,1,0,3)), P),(2,1,0,3))
            _Hess4U4 = np.transpose(np.dot(np.transpose(_Hess4U3,(3,1,2,0)), P),(3,1,2,0))
            return _Hess4U4
        return Hess4U

    def get_T_Hamiltonian(self, phi_ext_s_0=0, phi_ext_l_0=0):
        def T(n, P=np.identity(3)):
            n = P.dot(n)
            (na, nb, nc) = (n[0], n[1], n[2])
            _T = 4*(self.ECa/hbar)*(na)**2 + \
                 4*(self.ECb/hbar)*(nb)**2 + \
                 4*(self.ECc/hbar)*(nc)**2 + \
                 4*(self.Ecoup/hbar)*(na-nc)**2
            return _T
        return T

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

    def get_U_matrix(self, phi_ext_s_0=0, phi_ext_l_0=0, mode = 'analytical'):
        U = self.get_U(phi_ext_s_0=phi_ext_s_0,
                       phi_ext_l_0=phi_ext_l_0)
        if mode == 'analytical':
            x0 = np.array([0, 0, 0])
            res = minimize(U, x0, method='SLSQP', tol=1e-12)
            HessU = self.get_HessU(phi_ext_s_0=phi_ext_s_0,
                           phi_ext_l_0=phi_ext_l_0)
            quad = res.x, HessU([res.x[0], res.x[1], res.x[2]])/2
        else:
            quad = self.get_quadratic_form(U)
        return quad

    def get_T_matrix(self, phi_ext_s_0=0, phi_ext_l_0=0, mode = 'analytical'):
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

    def get_freqs_kerrs(self, phi_ext_s_0=0, phi_ext_l_0=0):
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

        ZPF = popt2**(-1./4)

        Xi2 = popt2*(ZPF**2)/2/np.pi
        Xi3 = popt3*(ZPF**3)/2/np.pi
        Xi4 = popt4*(ZPF**4)/2/np.pi

        # Plot to check fits to polynomial expansion
#        fig, ax = plt.subplots()
#        ax.plot(xVec, UxVec, label='UxVec')
#        ax.plot(xVec, popt_x[-1]+popt_x[-2]*xVec+popt_x[-3]*xVec**2,
#                label='fit2')
#        ax.plot(xVec, UxVec - (popt_x[-1]+popt_x[-2]*xVec+popt_x[-3]*xVec**2))
#        ax.plot(xVec, popt_x[-4]*xVec**3, label='fit3')
#        ax.plot(xVec, popt_x[-4]*xVec**3+popt_x[-5]*xVec**4, label='fit4')
#        ax.legend()
        return res1, res2, fs, Xi2, Xi3, Xi4, coeff2

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
