# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:55:50 2019

@author: Zaki
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

alpha = 2
chiTs = [np.pi, np.pi/2, np.pi/4]
r = np.linspace(-4*alpha, 4*alpha, 1001)
    
fig, ax = plt.subplots(len(chiTs))

for ii, chiT in enumerate(chiTs):
    X, Y = [], []
    
    for _r in r:
        ss = (_r**2-alpha**2)
        x = 2*np.exp(ss*(np.cos(chiT)-1))*np.cos(ss*np.sin(chiT))
        y1 = np.exp(ss*(np.cos(chiT)-1)+2*_r*alpha*np.sin(chiT))
        y1 *= np.cos(ss*np.sin(chiT)-2*_r*alpha*(np.cos(chiT)-1))
        y2 = np.exp(ss*(np.cos(chiT)-1)-2*_r*alpha*np.sin(chiT))
        y2 *= np.cos(ss*np.sin(chiT)+2*_r*alpha*(np.cos(chiT)-1))
        y = y1 + y2
        
        X.append(x)
        Y.append(y)
        
    ax[ii].set_title('pi/(chi*T) = %s' % str(round(np.pi/chiT)))
    ax[ii].plot(r, X, label='real')
    ax[ii].plot(r, Y, label='imag')
    ax[ii].set_ylim([-5000, 7000])
    ax[ii].legend()