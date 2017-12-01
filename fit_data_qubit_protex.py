# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:11:49 2016

@author: leghtas
"""

import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import circuit
import scipy.linalg as sl
import numpy.linalg as nl
import numdifftools as nd
from scipy.misc import factorial
from lmfit import minimize, Parameters
import scipy.ndimage.filters as flt
import matplotlib

plt.close('all')
Phi0 = sc.value('mag. flux quantum')
e = sc.elementary_charge
phi0 = Phi0/2/np.pi  # Phi_0=h/(2*e)
pi = sc.pi
hbar = sc.hbar
h = sc.h

wa = 2*pi*4e9  # res frequency (Hz/rad) HFSS 6.13
wb = 2*pi*6.2e9  # res frequency (Hz/rad) HFSS 3.31
wc = 2*pi*7.7045e9  # res frequency (Hz/rad) HFSS 7.3
Za = 130.  # res impedance in Ohm (cavity)
Zb = 130.  # res impedance in Ohm (qubit)
Zc = 50.
LJ1 = 1.0 * 15e-9  # Josephson inductance, each junction has 2*LJ
LJ2 = 1.1 * 15e-9 * 1.0
# ECa = h*200*1e6
# EJa = 40*ECa
# ELa = EJa/1e4
# wa, Za, LJ = circuit.get_w_Z_LJ_from_E(EC, EJ, EL)

na, nb, nc = 6, 6, 6

phi_ext_sweep = np.linspace(pi-2*pi/10, pi+2*pi/10, 101)
phiVec = np.linspace(-4*0.5*2*pi, 4*0.5*2*pi, 101)
ng_sweep = np.linspace(-1, 1, 21)

f = np.zeros((3, len(phi_ext_sweep)))
k = np.zeros((3, len(phi_ext_sweep)))
Xi3 = np.zeros((5, len(phi_ext_sweep)))
Xi4 = np.zeros((5, len(phi_ext_sweep)))
resx = np.zeros(len(phi_ext_sweep))
resy = np.zeros(len(phi_ext_sweep))
resz = np.zeros(len(phi_ext_sweep))
gradUval = np.zeros(len(phi_ext_sweep))
gradUval_ = np.zeros(len(phi_ext_sweep))

ls = 20    
matplotlib.rcParams['xtick.labelsize'] = ls
matplotlib.rcParams['ytick.labelsize'] = ls
fs=30

if 1==1:
    def get_freqs_from_vals(x, params):
        print('eval')
        f = np.zeros((3, len(x)))
        
        wa = params['wa']
        wb = params['wb']
        wc = params['wc']
        LJ1 = params['LJ1']
        LJ2 = params['LJ2']
        SL = params['SL']
        BL = params['BL']
        SL0 = params['SL0']
        BL0 = params['BL0']
        Ecoup = params['Ecoup']
        Zb = params['Zb']
        Za = params['Za']

        ECa, ELa, EJ1 = circuit.get_E_from_w(wa, Za, LJ1)
        ECb, ELb, EJ2 = circuit.get_E_from_w(wb, Zb, LJ2)
        ECc, ELc, EJ = circuit.get_E_from_w(wc, Zc, LJ1)
        c = circuit.Circuit(ECa, ELa, ECb, ELb, EJ1, EJ2, na, nb,
                    ECc=ECc, ELc=ELc, Ecoup=Ecoup, nc=nc,
                    printParams=False)
        
        for kk, xx in enumerate(x):
            fs = c.get_freqs_only(phi_ext_s_0=SL*(xx+SL0), phi_ext_l_0=BL*(xx+BL0))
            f[:, kk] = np.sort(fs)
        return f
    def get_fa(x, params):
         f = get_freqs_from_vals(x, params)
         return f[0]
    def get_fb(x, params):
         f = get_freqs_from_vals(x, params)
         return f[1]


if 1==1:

    ### data ###
    folder = r'data/qubit_protex/'

    filename_buffer = r'spec_VNA_in3outB_sweepDC_follow_spec_mem_004_spec_buff2.dat.npy'
    buffer_data = np.load(folder+filename_buffer)
    buffer_flux = buffer_data[0]
    buffer_freq = buffer_data[1]

    filename_mem = r'spec_VNA_in3outB_sweepDC_follow_spec_mem_004_spec_mem2.dat.npy'
    mem_data = np.load(folder+filename_mem)
    mem_flux = mem_data[0]
    mem_freq = mem_data[1]

    fig, ax = plt.subplots(2)
    ax[0].plot(mem_flux, mem_freq/1e9, 'o')
    ax[1].plot(buffer_flux, buffer_freq/1e9, 'o')
    ax[1].set_xlabel('DC (V)')
    ax[0].set_ylabel('mem freq (GHz)')
    ax[1].set_ylabel('buff freq (GHz)')


if 1==1:
    def residualb(params, x, data):
        model = get_fb(x, params)
        return (data-model)
    
    params = Parameters()
    params.add('wa', value=outb.params['wa'], vary=False)
    params.add('wb', value=outb.params['wb'], vary=True)
    params.add('wc', value=outb.params['wc'], vary=False)
    params.add('Za', value=Za, vary=False)
    params.add('Zb', value=outb.params['Zb'], vary=True)
#    params.add('LJ1', value=0.8*LJ1)
#    params.add('LJ2', value=0.8*LJ1)
    params.add('LJ1', value=outb.params['LJ1'], vary=True)
    params.add('LJ2', value=outb.params['LJ1'], expr='LJ1', vary=True)
    params.add('SL', value=outb.params['SL'], vary=True)
    params.add('BL', value=outb.params['BL'], vary=True)
    params.add('SL0', value=outb.params['SL0'], vary=True)
    params.add('BL0', value=outb.params['BL0'], vary=True)
    params.add('Ecoup', value=outb.params['Ecoup'], vary=False)

    
    data = buffer_freq
    x = buffer_flux
    
#    xhat = np.fft.rfftfreq(len(x),x[1]-x[0])
#    data_fft = np.fft.rfft(data-np.mean(data))
#    fig, ax= plt.subplots()
#    ax.plot(xhat, np.abs(data_fft))
    
    outb = minimize(residualb, params, args=(x, data))

    fig, ax = plt.subplots(figsize=(16,12))
    ax.plot(buffer_flux, buffer_freq/1e9,'o', label='data')
    #ax.plot(x, get_fb(x, params)/1e9, label='guess')
    ax.plot(x, get_fb(x, outb.params)/1e9,linewidth=2.0, label='theory')
    ax.set_xlabel('Voltage (V)', fontsize=fs)
    ax.set_ylabel('Frequency (GHz)', fontsize=fs)
    ax.set_title('Buffer mode spectroscopy', fontsize=fs)
    ax.legend(fontsize=fs)
    plt.savefig('spec_buff.pdf')
    
if 1==1:
    def residuala(params, x, data):
        model = get_fa(x, params)
        return (data-model)
    
    params = Parameters()
    params.add('wa', value=outb.params['wa']+2*np.pi*100e6, vary=True)
    params.add('wb', value=outb.params['wb'], vary=False)
    params.add('wc', value=outb.params['wc'], vary=False)
    params.add('Zb', value=outb.params['Zb'], vary=False)
    params.add('Za', value=Za, vary=True)
#    params.add('LJ1', value=0.8*LJ1)
#    params.add('LJ2', value=0.8*LJ1)
    params.add('LJ1', value=outb.params['LJ1'], vary=False)
    params.add('LJ2', value=outb.params['LJ2'], expr='LJ1', vary=False)
    params.add('SL', value=outb.params['SL'], vary=False)
    params.add('BL', value=outb.params['BL'], vary=False)
    params.add('SL0', value=outb.params['SL0'], vary=False)
    params.add('BL0', value=outb.params['BL0'], vary=False)
    params.add('Ecoup', value=outb.params['Ecoup'], vary=False)

    data = mem_freq
    x = mem_flux
    
    outa = minimize(residuala, params, args=(x, data))

    fig, ax = plt.subplots(figsize=(16,12))
    ax.plot(x, data/1e9,'o', label='data')
#    ax.plot(x, get_fa(x, params)/1e9, label='guess')
    ax.plot(x, get_fa(x, outa.params)/1e9,linewidth=2.0, label='theory')
    ax.set_xlabel('Voltage (V)', fontsize=fs)
    ax.set_ylabel('Frequency (GHz)', fontsize=fs)
    ax.set_title('Memory mode spectroscopy', fontsize=fs)
    ax.legend(fontsize=fs)
    plt.savefig('spec_mem.pdf')

if 1==1:
    fig, ax = plt.subplots(2,figsize=(18,20))
    ax[0].plot(buffer_flux, buffer_freq/1e9,'o', label='data')
#    ax.plot(x, get_fa(x, params)/1e9, label='guess')
    ax[0].plot(buffer_flux, get_fb(buffer_flux, outb.params)/1e9,linewidth=2.0, label='theory')
    ax[0].set_ylabel('Frequency (GHz)', fontsize=fs)
    ax[0].set_title('Buffer mode spectroscopy', fontsize=fs)
    ax[0].legend(fontsize=fs)
    
    _mem_flux = np.linspace(mem_flux[0], mem_flux[-1], 10*len(mem_flux))
    ax[1].plot(mem_flux, mem_freq/1e9,'o')
#    ax.plot(x, get_fa(x, params)/1e9, label='guess')
    ax[1].plot(_mem_flux, get_fa(_mem_flux, outa.params)/1e9,linewidth=2.0)
    ax[1].set_xlabel('Voltage (V)', fontsize=fs)
    ax[1].set_ylabel('Frequency (GHz)', fontsize=fs)
    ax[1].set_title('Memory mode spectroscopy', fontsize=fs)
    ax[1].legend(fontsize=fs)
    plt.savefig('spec_buff_mem.pdf')
    
if 1==0:
    print('wa/2pi = %s GHz' % float(outa.params['wa'].value/2/np.pi/1e9))
    print('wb/2pi = %s GHz' % float(outb.params['wb'].value/2/np.pi/1e9))
    print('wc/2pi = %s GHz' % float(outc.params['wc'].value/2/np.pi/1e9))
    print('LJ per junction = %s nH' % float(outb.params['LJ1'].value/1e-9))
    print('Eccoup/h = %s GHz' % float(outc.params['Ecoup'].value/h/1e9))