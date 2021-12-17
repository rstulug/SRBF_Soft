#!/usr/bin/env pypy3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 13:21:31 2018

@author: Rasit ULUG
Middle East Technical University
"""

import numpy as np
from numba import jit

"""
Some function converted from Graflab software

Bucha, B., & Janák, J. (2013). A MATLAB-based graphical user interface program 
for computing functionals of the geopotential up to ultra-high degrees and orders. 
Computers & geosciences, 56, 186-196.

"""
    
@jit
def legendre(n,fi):
    """
    Computes Fully Normalized Associated Legendre Polynomials
    
    n = degree
    fi = angle(degree) between equatorial plane to pole
    
    Rasit ULUG 25.12.2018
    Middle East Technical University
    """
    m = n
    fi = fi*np.pi/180
    pnm  = np.zeros((n+1,m+1))
    dpnm = np.zeros((n+1,m+1))
    
    pnm[0,0]  = 1
    dpnm[0,0] = 0
    pnm[1,1]  = np.sqrt(3)*np.cos(fi)
    dpnm[1,1] = -np.sqrt(3)*np.sin(fi)
    
    for i in range(2,n+1):
        pnm[i,i] = np.sqrt((2*i+1)/(2*i))*np.cos(fi)*pnm[i-1,i-1]
        dpnm[i,i] = np.sqrt((2*i+1)/(2*i))*((np.cos(fi)*dpnm[i-1,i-1] - np.sin(fi)*pnm[i-1,i-1]))
    for i in range(n):
        pnm[i+1,i] = np.sqrt(2*(i+1)+1)*np.sin(fi)*pnm[i,i]
        dpnm[i+1,i] = np.sqrt(2*(i+1)+1)*((np.cos(fi)*pnm[i,i])+(np.sin(fi)*dpnm[i,i]))
    j = 0
    k = 2
    while(1):
        for i in range(k,n+1):
            pnm[i,j] = np.sqrt((2*i+1)/((i-j)*(i+j)))*((np.sqrt(2*i-1)*np.sin(fi)*pnm[i-1,j]) -\
               (np.sqrt(((i+j-1)*(i-j-1))/(2*i-3))*pnm[i-2,j]))
            
            dpnm[i,j] = np.sqrt((2*i+1)/((i-j)*(i+j)))*((np.sqrt(2*i-1)*np.sin(fi)*dpnm[i-1,j]) +\
                (np.sqrt(2*i-1)*np.cos(fi)*pnm[i-1,j])-(np.sqrt(((i+j-1)*(i-j-1))/(2*i-3))*dpnm[i-2,j]))
        j += 1
        k += 1
        if j>m:
            break
    return(pnm,dpnm)
     
def normal_gravity80(fi,mode=0):
    """
    Computes normal gravity (on ellipsoid) wrt GRS80 ellipsoid
    
    Rasit ULUG 25.12.2015
    Middle East Technical University
    
    """
    fi = fi * np.pi/180
    if mode == 0:
        a = 6378137.0
        b = 6356752.3141
        na = 9.7803267715
        nb = 9.8321863685
        normal = (a*na*np.cos(fi)**2 + b*nb*np.sin(fi)**2)/ (np.sqrt(a**2 * np.cos(fi)**2 + b**2 * np.sin(fi)**2))
    else:
        normal = 9.780326772 *(1+0.0052790414*np.sin(fi)**2 + 0.0000232718*np.sin(fi)**4 + 0.0000001262*np.sin(fi)**6 + 0.0000000007* np.sin(fi)**8)
    return(normal)

@jit
def disturbing_potential(args):
    """
    Computes disturing  potential at given location
    C,S = fully normalized Stokes coefficients
    n = degree 
    r = location vector
    GM gravity constant of earth (can obtained form GGM)
    radius = mean radius of earth (can obtained from GGM)
    
    Rasit ULUG 25.12.2015
    Middle East Technical University
    """
    C = args[0];S = args[1];n = args[2]
    r = args[3]; GM=args[4]; radius= args[5];kind=args[6]

    size = np.shape(r)[0]
    potential = np.zeros(size)
    for k in range(size):
        dist = np.sqrt(r[k,0]**2+r[k,1]**2+r[k,2]**2)
        if kind == 0:
            lat = np.arcsin(r[k,2]/dist)
        else:
            lat = np.arctan(r[k,2]/np.sqrt(r[k,0]**2+r[k,1]**2))
        long = np.arctan2(r[k,1],r[k,0])
        
        pnm = legendre(n,lat*180/np.pi)[0]
        pot_last = 0
        for i in range(n+1):
            pot_first = 0
            const = ((radius/dist)**(i))
            for m in range(i+1):
                pot_first += pnm[i,m]*(C[i,m]*np.cos(m*long)+S[i,m]*np.sin(m*long))
            pot_last += pot_first*const
        potential[k] = pot_last*(GM/dist)
    return(potential)

@jit
def anomaly_from_ggm(args):
    """
    Computes gravity anomaly at given location 
    C,S = fully normalized Stokes coefficients
    n = degree 
    r = location vector
    GM gravity constant of earth (can obtained form GGM)
    radius = mean radius of earth (can obtained from GGM)
    
    Rasit ULUG 25.12.2015
    Middle East Technical University
    """
    C = args[0];S = args[1];n = args[2]
    r = args[3]; GM=args[4]; radius= args[5]; kind=args[6]

    size = np.shape(r)[0]
    anomaly = np.zeros(size)
    for k in range(size):
        dist = np.sqrt(r[k,0]**2+r[k,1]**2+r[k,2]**2)
        if kind == 0:
            lat = np.arcsin(r[k,2]/dist)
        else:
            lat = np.arctan(r[k,2]/np.sqrt(r[k,0]**2+r[k,1]**2))
        long = np.arctan2(r[k,1],r[k,0])
        long = np.arctan2(r[k,1],r[k,0])
        
        pnm = legendre(n,lat*180/np.pi)[0]
        anom_last = 0
        for i in range(n+1):
            anom_first = 0
            const = ((radius/dist)**(i))*((i-1)/dist)
            for m in range(i+1):
                anom_first += pnm[i,m]*(C[i,m]*np.cos(m*long)+S[i,m]*np.sin(m*long))
            anom_last += anom_first*const
        anomaly[k] = anom_last*(GM/dist)
    return(anomaly)

@jit
def disturbance_from_ggm(args):
    """
    Computes gravity disturbance at given location 
    C,S = fully normalized Stokes coefficients
    n = degree 
    r = location vector
    GM gravity constant of earth (can obtained form GGM)
    radius = mean radius of earth (can obtained from GGM)
    
    Rasit ULUG 25.12.2015
    Middle East Technical University
    """
    C = args[0];S = args[1];n = args[2]
    r = args[3]; GM=args[4]; radius= args[5]; kind=args[6]

    size = np.shape(r)[0]
    disturbance = np.zeros(size)
    for k in range(size):
        dist = np.sqrt(r[k,0]**2+r[k,1]**2+r[k,2]**2)
        if kind == 0:
            lat = np.arcsin(r[k,2]/dist)
        else:
            lat = np.arctan(r[k,2]/np.sqrt(r[k,0]**2+r[k,1]**2))
        long = np.arctan2(r[k,1],r[k,0])
        long = np.arctan2(r[k,1],r[k,0])
        
        pnm = legendre(n,lat*180/np.pi)[0]
        dist_last = 0
        for i in range(n+1):
            dist_first = 0
            const = ((radius/dist)**(i))*((i+1)/dist)
            for m in range(i+1):
                dist_first += pnm[i,m]*(C[i,m]*np.cos(m*long)+S[i,m]*np.sin(m*long))
            dist_last += dist_first*const
        disturbance[k] = dist_last*(GM/dist)
    return(disturbance)
