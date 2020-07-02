# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:19:52 2019

@author: Zaki
"""

import matplotlib.pyplot as plt
import numpy as np
import circuit as cc

plt.close('all')
A = np.array([[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 12]])

I = [1, 2, 3, 4]
Q = [1, 2, 3]

u = 0
B = np.zeros((3, 4))

for qq, q in enumerate(Q):
    for ii, i in enumerate(I):
        u += 1
        B[qq, ii] = u
        
fig, ax = plt.subplots(2)
cc.pcolor_z(ax[0], I, Q, A)
cc.pcolor_z(ax[1], I, Q, B)