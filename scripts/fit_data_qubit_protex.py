# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:11:49 2016

@author: leghtas
"""

import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
from circuit_odd_coupling import *
import scipy.linalg as sl
import numpy.linalg as nl
import numdifftools as nd
from scipy.misc import factorial
from lmfit import minimize, Parameters
import scipy.ndimage.filters as flt
import matplotlib
import matplotlib.gridspec as gridspec


plt.close('all')
iffit = False

wa = 2*pi*4e9  # res frequency (Hz/rad) HFSS 6.13
wb = 2*pi*6.2e9  # res frequency (Hz/rad) HFSS 3.31
wc = 2*pi*7.7045e9  # res frequency (Hz/rad) HFSS 7.3
Za = 130.  # res impedance in Ohm (cavity)
Zb = 100.  # res impedance in Ohm (qubit)
Zc = 50.
LJ1 = 1.0 * 15e-9  # Josephson inductance, each junction has 2*LJ
LJ2 = 1.1 * 15e-9 * 1.0
# ECa = h*200*1e6
# EJa = 40*ECa
# ELa = EJa/1e4
# wa, Za, LJ = circuit.get_w_Z_LJ_from_E(EC, EJ, EL)

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
fs = 30

wa = 26253111758
wb = 38405084434
wc = 96817602398
Zb = 100
Za = 127
LJ1 = 1.32e-08
LJ2 = 1.32e-08
SL = 1.453
BL = 18.68
SL0 = 2.4549
BL0 = 0.127
Ecoup = 9e-10


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

        ECa, ELa, EJ1 = get_E_from_w(wa, Za, LJ1)
        ECb, ELb, EJ2 = get_E_from_w(wb, Zb, LJ2)
        ECc, ELc, EJ = get_E_from_w(wc, Zc, LJ1)
        c = CircuitOddCoupling(ECa, ELa, ECb, ELb, EJ1, EJ2,
                               ECc=ECc, ELc=ELc, Ecoup=Ecoup,
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

    def residuala(params, x, data):
        model = get_fa(x, params)
        return (data-model)
        
    def residualb(params, x, data):
        model = get_fb(x, params)
        return (data-model)

if 1==1: ## LOAD DATA ##
    folder_root = r'../data/qubit_protex/'
    folder_data = folder_root + r'analyzed_data/'

    filename_buffer = r'spec_VNA_in3outB_sweepDC_follow_spec_mem_004_spec_buff2.dat.npy'
    buffer_data = np.load(folder_data+filename_buffer)
    buffer_flux = buffer_data[0]
    buffer_freq = buffer_data[1]
    
    buffer_flux = np.concatenate((buffer_data[0][40:80],buffer_data[0][255:290]))
    buffer_freq = np.concatenate((buffer_data[1][40:80],buffer_data[1][255:290]))

    filename_mem = r'spec_VNA_in3outB_sweepDC_follow_spec_mem_004_spec_mem2.dat.npy'
    mem_data = np.load(folder_data+filename_mem)
    mem_flux = mem_data[0]
    mem_freq = mem_data[1]
#    
    mem_flux = np.concatenate((mem_data[0][31:64],mem_data[0][198:225]))
    mem_freq = np.concatenate((mem_data[1][31:64],mem_data[1][198:225]))

    if 1==1:
        fig = plt.figure(figsize = (15,15))
        gs = gridspec.GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax1.plot(mem_flux, mem_freq/1e9, 'o')
        ax2.plot(buffer_flux, buffer_freq/1e9, 'o')
        ax2.set_xlabel('DC (V)')
        ax1.set_ylabel('mem freq (GHz)')
        ax2.set_ylabel('buff freq (GHz)')
        ax3 = fig.add_subplot(gs[:, 1])
        jj = 0
        mem_freqs = []
        buffer_freqs = []
        for ii in range(len(buffer_flux)):
            if jj<len(mem_flux) and mem_flux[jj]==buffer_flux[ii]:
                mem_freqs.append(mem_freq[jj]/1e9)
                buffer_freqs.append(buffer_freq[ii]/1e9)
                jj+=1
                
        mem_freqs = [x for _,x in sorted(zip(buffer_freqs,mem_freqs))]
        buffer_freqs = [x for _,x in sorted(zip(buffer_freqs,buffer_freqs))]
        ax3.scatter(buffer_freqs, mem_freqs)
        
        coefs = np.polyfit(buffer_freqs, mem_freqs, 3)
        freqs = np.linspace(min(buffer_freqs), max(buffer_freqs), 50)
        ax3.plot(freqs, np.polyval(coefs, freqs))
        ax3.set_xlabel('Memory frequency')
        ax3.set_ylabel('Buffer frequency')    
               


if 1==1: # FIT BUFFER
    folder_img = folder_root + r'figures/'
    params = Parameters()
    
    wa = 26253111758
    wb = 38407337530
    wc = 96817602398
    Zb = 100
    Za = 127
    LJ1 = 1.318e-08
    LJ2 = 1.318e-08
    SL = 1.47
    BL = 18.85
    SL0 = 2.41
    BL0 = 0.12084
    Ecoup = 9e-10
    
    wa = 26294488130
    wb = 38398830335
    wc = 96817602398
    Zb = 76
    Za = 98
    LJ1 = 9.822e-09
    LJ2 = 9.7489e-09
    SL = 1.4709
    BL = 18.804
    SL0 = 2.4130
    BL0 = 0.12590
    Ecoup = 9e-10
    
    params.add('wa', value=wa, vary=False)
    params.add('wb', value=wb, vary=False)
    params.add('wc', value=wc, vary=False)
    params.add('Zb', Zb, vary=False)
    params.add('Za', Za, vary=False)
    params.add('LJ1', value=LJ1, vary=True)
    params.add('LJ2', value=LJ2, vary = True)#expr='LJ1', vary=True)
    params.add('SL', value=SL, vary=True)
    params.add('BL', value=BL, vary=True)
    params.add('SL0', value=SL0, vary=True)
    params.add('BL0', value=BL0, vary=True)
    params.add('Ecoup', value=Ecoup, vary=False)

    data = buffer_freq
    x = buffer_flux
    
    if iffit:
        outb = minimize(residualb, params, args=(x, data))

    fig, ax = plt.subplots(figsize=(16,12))
    ax.plot(buffer_flux, buffer_freq/1e9,'o', label='data')
    if iffit:
        ax.plot(x, get_fb(x, outb.params)/1e9,linewidth=2.0, label='theory')
    else:
        ax.plot(x, get_fb(x, params)/1e9, label='guess')
#    ax.plot([min(buffer_flux), max(buffer_flux)], outb.params['wb']/2/np.pi/1e9*np.ones((2)), label = 'wb')
    ax.set_xlabel('Voltage (V)', fontsize=fs)
    ax.set_ylabel('Frequency (GHz)', fontsize=fs)
    ax.set_title('Buffer mode spectroscopy', fontsize=fs)
    ax.legend(fontsize=fs)
    plt.savefig(folder_img+'spec_buff.pdf')
    
if 1==1: # FIT MEMORY #
    folder_img = folder_root + r'figures/'
    params = Parameters()
    params.add('wa', value=wa, vary=True)
    params.add('wb', value=outb.params['wb'] if iffit else wb, vary=False)
    params.add('wc', value=outb.params['wc'] if iffit else wc, vary=False)
    params.add('Zb', value=outb.params['Zb'] if iffit else Zb, vary=False)
    params.add('Za', Za, vary=True)
    params.add('LJ1', value=outb.params['LJ1'] if iffit else LJ1, vary=False)
    params.add('LJ2', value=outb.params['LJ2'] if iffit else LJ2, expr='LJ1', vary=False)
    params.add('SL', value=outb.params['SL'] if iffit else SL, vary=False)
    params.add('BL', value=outb.params['BL'] if iffit else BL, vary=False)
    params.add('SL0', value=outb.params['SL0'] if iffit else SL0, vary=False)
    params.add('BL0', value=outb.params['BL0'] if iffit else BL0, vary=False)
    params.add('Ecoup', value=outb.params['Ecoup'] if iffit else Ecoup, vary=False)

    data = mem_freq
    x = mem_flux
    
    if iffit:
        outa = minimize(residuala, params, args=(x, data))

    fig, ax = plt.subplots(figsize=(16,12))
    ax.plot(x, data/1e9,'o', label='data')

    if iffit:
        ax.plot(x, get_fa(x, outa.params)/1e9,linewidth=2.0, label='theory')
    else:
        ax.plot(x, get_fa(x, params)/1e9, label='guess')
        
    ax.set_xlabel('Voltage (V)', fontsize=fs)
    ax.set_ylabel('Frequency (GHz)', fontsize=fs)
    ax.set_title('Memory mode spectroscopy', fontsize=fs)
    ax.legend(fontsize=fs)
    plt.savefig(folder_img+'spec_mem.pdf')

if 1==1: # PLOT BOTH ON SAME PLOT
    folder_img = folder_root + r'figures/'
    fig, ax = plt.subplots(2,figsize=(18,20))
    ax[0].scatter(buffer_flux*3e-2, buffer_freq/1e9,marker = 'o', color = 'green', edgecolors = 'green', label='data')
#    ax.plot(x, get_fa(x, params)/1e9, label='guess')
    if 'outb' in locals():
        ax[0].plot(buffer_flux*3e-2, get_fb(buffer_flux, outb.params)/1e9, color = 'gray', linewidth=2.0, label='theory')
    ax[0].set_ylabel('Frequency (GHz)', fontsize=fs)
    ax[0].set_title('Buffer mode spectroscopy', fontsize=fs)
    ax[0].legend(fontsize=fs)
    
    _mem_flux = np.linspace(mem_flux[0], mem_flux[-1], 10*len(mem_flux))
    ax[1].scatter(mem_flux*3e-2, mem_freq/1e9,marker = 'o', color = 'r', edgecolors = 'red')
#    ax.plot(x, get_fa(x, params)/1e9, label='guess')
    if 'outb' in locals():
        ax[1].plot(_mem_flux*3e-2, get_fa(_mem_flux, outb.params)/1e9, color = 'grey',linewidth=2.0)
    ax[1].set_xlabel('Current (mA)', fontsize=fs)
    ax[1].set_ylabel('Frequency (GHz)', fontsize=fs)
    ax[1].set_title('Memory mode spectroscopy', fontsize=fs)
    ax[1].legend(fontsize=fs)
    plt.savefig(folder_img + 'spec_buff_mem.pdf')

if 1==0: # EFFECT on wa=f(wb)
    fig, ax = plt.subplots(1, figsize =(12, 12))
    fluxes = np.linspace(-10,10, 1000)
    
    wa = 26294488130
    wb = 38398830335
    wc = 96817602398
    Zb = 76
    Za = 98
    LJ1 = 9.822e-09
    LJ2 = 9.7489e-09
    SL = 1.4709
    BL = 18.804
    SL0 = 2.4130
    BL0 = 0.12590
    Ecoup = 9e-10
    
    params.add('wa', value=wa, vary=False)
    params.add('wb', value=wb, vary=False)
    params.add('wc', value=wc, vary=False)
    params.add('Zb', Zb, vary=False)
    params.add('Za', Za, vary=False)
    params.add('LJ1', value=LJ1, vary=True)
    params.add('LJ2', value=LJ2, vary = True)#expr='LJ1', vary=True)
    params.add('SL', value=SL, vary=True)
    params.add('BL', value=BL, vary=True)
    params.add('SL0', value=SL0, vary=True)
    params.add('BL0', value=BL0, vary=True)
    params.add('Ecoup', value=Ecoup, vary=False)
    
    for factor in np.linspace(0.1, 1, 5):
        params.add('Zb', value=Zb*factor, vary=True)
        params.add('Za', value=Za*factor, vary=True)
        fs = get_freqs_from_vals(fluxes, params)/1e9
        fb = fs[1]
        fa = fs[0]
        ax.scatter(fb, fa, 10)
        ax.scatter([min(fb), max(fb)], [min(fa), max(fa)], 30)
#    ax.scatter(buffer_freqs, mem_freqs)
    
    
if 1==0: # PRINT
    print('wa/2pi = %s GHz' % float(outa.params['wa'].value/2/np.pi/1e9))
    print('wb/2pi = %s GHz' % float(outb.params['wb'].value/2/np.pi/1e9))
    print('LJ (total at 0 flux) = %s nH' % float(outb.params['LJ1'].value/1e-9))
    print('Za = %s Ohm' % float(outa.params['Za'].value))
    print('Zb = %s Ohm' % float(outa.params['Zb'].value))
    
if 1==0: # sweep LJ error
    params = Parameters()
    params.add('wa', value=outa.params['wa'])
    params.add('wb', value=outb.params['wb'])
    params.add('wc', value=outb.params['wc'])
    params.add('Zb', value=outb.params['Zb'], vary=True)
    params.add('Za', value=outa.params['Za'], vary=False)
    params.add('SL', value=outb.params['SL'])
    params.add('BL', value=outb.params['BL'])
    params.add('SL0', value=outb.params['SL0'], vary=True)
    params.add('Ecoup', value=outb.params['Ecoup'])

    data = buffer_freq
    x = buffer_flux
    fig, ax = plt.subplots(figsize=(16,12))

    dLJ_rel_Vec = np.linspace(0.01, 0.1,1)
    
    for dLJ_rel in dLJ_rel_Vec:
        print(dLJ_rel)
        params.add('LJ1', value=outb.params['LJ1']*(1+dLJ_rel/2.), vary=False)
        params.add('LJ2', value=outb.params['LJ1']*(1-dLJ_rel/2.), vary=False)
        for BL0 in np.linspace(0,0.2,5):
        # outb = minimize(residualb, params, args=(x, data))
            params.add('BL0', value=BL0, vary=True)
            ax.plot(x, get_fb(x, params)/1e9)

    ax.plot(buffer_flux, buffer_freq/1e9,'o', label='data')
    ax.set_xlabel('Voltage (V)', fontsize=fs)
    ax.set_ylabel('Frequency (GHz)', fontsize=fs)
    ax.set_title('Buffer mode spectroscopy', fontsize=fs)
    ax.legend(fontsize=fs)
    
    