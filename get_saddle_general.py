# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:37:27 2019

@author: Zaki
"""
import numpy as np
import matplotlib.pyplot as plt


def get_saddle(I1, I2, freq):
    I1m, I2m = np.meshgrid(I1, I2)
    
    fig, ax = plt.subplots()
    ax.plot(I1, freq.T)
    aa = np.min(freq, axis=1)
    _ia = np.argmin(freq, axis=1)
    _ib = np.argmax(aa)
    _ic = _ia[_ib]
    print(freq[_ib, _ic])
    
    tt = np.linspace(-0.1, 0.1, 11)
    
    def get_xyz(theta):
        x, y, z = [], [], []
        for t in tt:
            x.append(np.cos(theta)*t + I1[_ic])
            y.append(np.sin(theta)*t + I2[_ib])
            z.append(freq[_ib, _ic])
        return x, y, z
    
    thetaVecD = np.linspace(np.pi/2-0.5, np.pi/2+0.1, 21)
    thetaVecS = np.linspace(np.pi/2+0.1, np.pi/2+0.5, 21)

    def get_dist(thetaVec):
        dist = []
        for theta in thetaVec:
            x, y, z = get_xyz(theta)
            res = 0
            dist_vec = []
            for _t in np.arange(len(tt)):
                _x = np.min(np.abs(I1m-x[_t]) + np.abs(I2m-y[_t]) + np.abs(freq-z[_t]))
                dist_vec.append(_x)
                res += _x
            dist.append(res)
        return dist

    distD = get_dist(thetaVecD)
    distS = get_dist(thetaVecS)
    
    fig, ax = plt.subplots()
    ax.plot(thetaVecD-np.pi/2, distD)
    ax.plot(thetaVecS-np.pi/2, distS)
        
    xD, yD, zD = get_xyz(thetaVecD[np.argmin(distD)])
    xS, yS, zS = get_xyz(thetaVecS[np.argmin(distS)])
    fig, ax = plt.subplots()
    ax.pcolor(I1, I2, freq)
    ax.scatter(I1[_ic], I2[_ib], color='red')
    ax.scatter(xD, yD, zD, color='white')
    ax.scatter(xS, yS, zS, color='white')
    
    print('(i10 = %s, i20 = %s)' % (I1[_ic], I2[_ib]))
    print('theta_Delta = %s ' % (thetaVecD[np.argmin(distD)]-np.pi/2))
    print('theta_Sigma = %s ' % (thetaVecS[np.argmin(distS)]-np.pi/2))
    
if __name__ == '__main__':
    plt.close('all')
    I2 = np.load('I2.npy')
    I1 = np.load('I1.npy')
    freq = np.load('fitted_freq.npy')
    get_saddle(I1, I2, freq)
    