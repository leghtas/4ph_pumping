# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:30:59 2018

@author: checkhov
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, sproot, splev

plt.close('all')
phi = np.linspace(0,2*np.pi, 11)
pot = np.cos(phi)

spline = splrep(phi, pot)
def f(x):
    return splev(x, spline)
roots = sproot(spline)

phi2 = np.linspace(0,2*np.pi, 101)
fig, ax = plt.subplots(2)
ax[0].plot(phi, pot, 'o')
ax[0].plot(phi2, f(phi2), label='interpol')
ax[0].plot(phi2, np.cos(phi2), label='true')
ax[0].legend()
ax[1].plot(phi2, f(phi2)-np.cos(phi2), '.')