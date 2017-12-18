# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:02:22 2017

@author: checkhov
"""

import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.Symbol('x')
a = sp.Symbol('a')
print(a*x + x)

def f(x):
    return sp.sin(x)
def df(x):
    return sp.diff(f(x), x)
    
print(df(x))
print(df(x).subs(x, 0))
