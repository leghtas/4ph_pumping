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
    def get_fc(x, params):
         f = get_freqs_from_vals(x, params)
         return f[2]


if 1==0:
    fig, ax = plt.subplots(3, 3, figsize=(16,8))
#    ax[0].plot(phi_ext_sweep/np.pi, resx/np.pi)
#    ax[0].plot(phi_ext_sweep/np.pi, resy/np.pi)
#    ax[0].plot(phi_ext_sweep/np.pi, resz/np.pi)
    ax[0,0].plot(phi_ext_sweep/2/np.pi, f[0,:]/1e9)
    ax[1,0].plot(phi_ext_sweep/2/np.pi, f[1,:]/1e9)
    ax[2,0].plot(phi_ext_sweep/2/np.pi, f[2,:]/1e9)
    ax[0,0].plot(phi_ext_sweep/2/np.pi, k[0,:]/1e9)
    ax[1,0].plot(phi_ext_sweep/2/np.pi, k[1,:]/1e9)
    ax[2,0].plot(phi_ext_sweep/2/np.pi, k[2,:]/1e9)
    ax[0,1].plot(phi_ext_sweep/2/np.pi, Xi3[0,:]/1e6)
    ax[0,1].plot(phi_ext_sweep/2/np.pi, Xi3[2,:]/1e6)
    ax[1,1].plot(phi_ext_sweep/2/np.pi, Xi3[1,:]/1e6)
    ax[1,1].plot(phi_ext_sweep/2/np.pi, Xi3[3,:]/1e6)
    ax[0,2].plot(phi_ext_sweep/2/np.pi, Xi4[0,:]/1e6)
    ax[0,2].plot(phi_ext_sweep/2/np.pi, Xi4[2,:]/1e6)
    ax[1,2].plot(phi_ext_sweep/2/np.pi, Xi4[1,:]/1e6)
    ax[1,2].plot(phi_ext_sweep/2/np.pi, Xi4[3,:]/1e6)

    ax[0,0].set_title('frequency (GHz)')
    ax[0,1].set_title('Xi3 (MHz)')
    ax[0,2].set_title('Kerr (MHz)')
    ax[0,0].set_ylabel('MEMORY')
    ax[1,0].set_ylabel('BUFFER')
    ax[2,0].set_ylabel('READOUT')
    ax[2,0].set_xlabel('flux (/2pi)')
    ax[2,1].set_xlabel('flux (/2pi)')
    ax[2,2].set_xlabel('flux (/2pi)')

if 1==1:
    fig, ax = plt.subplots(3, figsize=(12,8), sharex=True)
    a=0.755e-3/3
    ax[0].plot(a*phi_ext_sweep/np.pi, f[0,:]/1e9)
    ax[1].plot(a*phi_ext_sweep/np.pi, f[1,:]/1e9)
    ax[2].plot(a*phi_ext_sweep/np.pi, f[2,:]/1e9)

    ### data ###
    folder = r'data/no_squid/analyzed_data/'

    filename_readout = r'sweep_DC_specVNA_DC2_in2outC_004.dat_'
    readout_freq = np.load(folder+filename_readout+'freq.npy')
    readout_flux = np.load(folder+filename_readout+'flux.npy')

    filename_buffer = r'sweep_DC_specVNA_DC2_in4outD_003.dat_'
    buffer_freq = np.load(folder+filename_buffer+'freq_fit.npy')
    buffer_flux = np.load(folder+filename_buffer+'flux.npy')

    filename_mem = r'VNA_sweep_DC_sweep_pump_freq_DC2_in2_outC_pump6_002.dat_'
    mem_freq = np.load(folder+filename_mem+'freq_fit.npy')
    mem_flux = np.load(folder+filename_mem+'flux.npy')

    offs = 1.2e-4
    ax[0].plot(mem_flux+1e-4-offs, mem_freq, 'o')
    ax[1].plot(buffer_flux+1e-4- offs, buffer_freq, 'o')
    ax[2].plot(readout_flux-offs, readout_freq/1e9, 'o')
    ax[2].set_xlabel('DC (V)')
    ax[0].set_ylabel('mem freq')
    ax[1].set_ylabel('buff freq')
    ax[2].set_ylabel('readout freq')

if 1==1: # fit
    def residualc(params, x, data):
        model = get_fc(x, params)
        return (data-model)
    
    params = Parameters()
    params.add('wa', value=wa, vary=False)
    params.add('wb', value=wb, vary=False)
    params.add('wc', value=wc-2*np.pi*95e6)
    params.add('LJ1', value=LJ1, vary=False)
    params.add('LJ2', value=LJ2, vary=False)
    params.add('SL', value=100, vary=False)
    params.add('BL', value=12e3)
    params.add('SL0', value=0, vary=False)
    params.add('BL0', value=-1e-4, vary=False)
    params.add('Ecoup', value=5e-25*2)
    
    x = readout_flux
    data = readout_freq
    outc = minimize(residualc, params, args=(x, data))
    fig, ax = plt.subplots()
    ax.plot(x, data, 'o', label='readout data')
#    ax.plot(x, get_fa(x, params), label='a')
#    ax.plot(x, get_fb(x, params), label='b')
    ax.plot(x, get_fc(x, outc.params), label='fit')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Frequency (Hz)')
    ax.legend()
    
if 1==1:
    def residualb(params, x, data):
        model = get_fb(x, params)
        return (data-model)
    
    params = Parameters()
    params.add('wa', value=wa, vary=False)
    params.add('wb', value=wb-2*np.pi*300e6)
    params.add('wc', value=outc.params['wc'], vary=False)
    params.add('LJ1', value=3*LJ1)
    params.add('LJ2', value=LJ1, expr='LJ1')
    params.add('SL', value=100, vary=False)
    params.add('BL', value=outc.params['BL'])
    params.add('SL0', value=0, vary=False)
    params.add('BL0', value=-2e-5, vary=False)
    params.add('Ecoup', value=outc.params['Ecoup'])

    exclude_arr = np.concatenate((np.arange(15, 30),
                                  np.arange(70, 85),
                                  np.arange(120, 135),
                                  np.arange(170, 190)
                                  ))
    data_filtered = []
    x_filtered = []
    for ii, ff in enumerate(buffer_freq):
        if ii not in exclude_arr:
            data_filtered.append(ff*1e9)
            x_filtered.append(buffer_flux[ii])
    
    
    data = data_filtered
    x = x_filtered
    outb = minimize(residualb, params, args=(x, data))
    fig, ax = plt.subplots()
    ax.plot(x_filtered, data_filtered, 'o')
    ax.plot(buffer_flux, buffer_freq*1e9, label='buffer data')
    ax.plot(x, get_fb(x, outb.params), label='fit')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Frequency (Hz)')
    ax.legend() 
    
if 1==1:
    def residuala(params, x, data):
        model = get_fa(x, params)
        return (data-model)
    
    params = Parameters()
    params.add('wa', value=wa-2*np.pi*700e6)
    params.add('wb', value=outb.params['wb'], vary=False)
    params.add('wc', value=outc.params['wc'], vary=False)
    params.add('LJ1', value=outb.params['LJ1'], vary=False)
    params.add('LJ2', value=LJ1, expr='LJ1', vary=False)
    params.add('SL', value=100, vary=False)
    params.add('BL', value=outc.params['BL'], vary=False)
    params.add('SL0', value=0, vary=False)
    params.add('BL0', value=-2e-5, vary=False)
    params.add('Ecoup', value=outc.params['Ecoup'], vary=False)

    exclude_arr = np.concatenate((np.arange(15, 30),
                                  np.arange(70, 85),
                                  np.arange(120, 135),
                                  np.arange(170, 190)
                                  ))
    data_filtered = []
    x_filtered = []
    for ii, ff in enumerate(buffer_freq):
        if ii not in exclude_arr:
            data_filtered.append(ff*1e9)
            x_filtered.append(buffer_flux[ii])
    
    
    data = mem_freq*1e9
    x = mem_flux
    outa = minimize(residuala, params, args=(x, data))
    fig, ax = plt.subplots()
    ax.plot(x, data, 'o', label='memory data')
    ax.plot(x, get_fa(x, outa.params), label='fit')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Frequency (Hz)')
    ax.legend() 

print('wa/2pi = %s GHz' % float(outa.params['wa'].value/2/np.pi/1e9))
print('wb/2pi = %s GHz' % float(outb.params['wb'].value/2/np.pi/1e9))
print('wc/2pi = %s GHz' % float(outc.params['wc'].value/2/np.pi/1e9))
print('LJ per junction = %s nH' % float(outb.params['LJ1'].value/1e-9))
print('Eccoup/h = %s GHz' % float(outc.params['Ecoup'].value/h/1e9))