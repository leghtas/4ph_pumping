# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:34:37 2019

@author: Zaki
"""

import matplotlib
import numpy as np

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)
    plt.show()


def connex_data(array, tol):
    shape = array.shape
    ret_array = np.ones(shape)*np.nan
    for jj in range(shape[0]):
        for ii in range(shape[1]):
            if array[jj, ii] is not np.nan:
                at_least = 0
                at_max = 0
                if ii>0:
                    at_max += 1
                    if not np.isnan(array[jj, ii-1]):
                        at_least+= 1
                if jj>0:
                    at_max += 1
                    if not np.isnan(array[jj-1, ii]) and jj>0:
                        at_least+= 1
                if ii<shape[1]-1:
                    at_max += 1
                    if not np.isnan(array[jj, ii+1]):
                        at_least+= 1
                if jj<shape[0]-1:
                    at_max += 1
                    if not np.isnan(array[jj+1, ii]):
                        at_least+= 1
                if (at_max-at_least)<4-tol:
                    ret_array[jj, ii] = array[jj, ii]
    return ret_array