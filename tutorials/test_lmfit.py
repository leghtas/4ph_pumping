#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:19:55 2017

@author: leghtas
"""

from lmfit import minimize, Parameters
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
def model(x, params):
    amp = params['amp']
    pshift = params['phase']
    freq = params['frequency']
    decay = params['decay']
    
    model = amp * np.sin(x * freq  + pshift) * np.exp(-x*x*decay)
    
    return model
    
def residual(params, x, data):
    _model = model(x, params)
    return (data-_model)

params = Parameters()
amp = 10
decay = 0.007
pshift = 0.2
freq = 3.0
params.add('amp', value=amp)
params.add('decay', value=decay)
params.add('phase', value=pshift)
params.add('frequency', value=freq)

x = np.linspace(0,10,401)
data = amp * np.sin(x * freq  + pshift) * np.exp(-x*x*decay)
data = data + np.random.randn(len(x))

out = minimize(residual, params, args=(x, data))

fig, ax = plt.subplots()
ax.plot(x, data, 'o')
ax.plot(x, model(x, out.params))