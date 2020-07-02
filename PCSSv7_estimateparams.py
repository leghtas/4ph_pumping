# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:06:28 2019

@author: Zaki
"""
import numpy as np
import scipy.constants as sc

Phi0 = sc.value('mag. flux quantum')
e = sc.elementary_charge
phi0 = Phi0/2/np.pi  # Phi_0=h/(2*e)
pi = sc.pi
hbar = sc.hbar
h = sc.h

wg = 2*np.pi*25.7e9
wb = 2*np.pi*4.88e9
LK = 4.25e-9

Csum = (1/LK)*(wg**2-wb**2)/(wg**2*wb**2)
LG = LK*wb**2/(wg**2-wb**2)

ELK = phi0**2/LK
ECsum = e**2/2/Csum

print('LK = %s nH' % str(LK*1e9))
print('Csum = %s fF' % str(Csum*1e15))
print('LG = %s nH' % str(LG*1e9))
print('ECsum = %s MHz' % str(ECsum/h/1e6))
print('ELK = %s GHz' % str(ELK/h/1e9))
print('ZG = %s Ohm' % str(np.sqrt(LG/Csum)))


print('wg/2pi = %s GHz' % str(1/np.sqrt(LG*Csum)/2/np.pi/1e9))
print('wb/2pi = %s GHz' % str(1/np.sqrt((LG+LK)*Csum)/2/np.pi/1e9))