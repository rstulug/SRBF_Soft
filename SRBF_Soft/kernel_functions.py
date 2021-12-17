#!/usr/bin/env pypy3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:54:22 2018

@author: Rasit ULUG
Middle East Technical University
"""

import numpy as np
from numba import jit,njit

"""
This file calculate:
    Point Mass Kernel
    Poisson Kernel
    Poisson Wavelet Kernel


The kernels can be found:

Tenzer, R., & Klees, R. (2008). The choice of the spherical radial basis 
functions in local gravity field modeling. Studia Geophysica et Geodaetica, 52(3), 287.    

"""

@njit 
def legendre(n,t):
    leg = np.zeros(n+1)
    leg[0] = 1
    leg[1] = t
    for i in range(2,n+1):
        leg[i] = (-(i-1)*leg[i-2] + (2*i-1)*t*leg[i-1])/i
    return(leg)

@jit
def pointmass_kernel(normx,normy,Lmin,sph_dist):
    kernel = 0
    lik   = np.sqrt(normx**2 + normy**2 - 2*normx*normy*np.cos(sph_dist))
    if Lmin != 0:
        leg = legendre(Lmin,np.cos(sph_dist))
        for i in range(0,Lmin):
            kernel += (1/(normy))*((normy/normx)**(i+1))*leg[i]
    result = (1/(lik)) - kernel
    return(result)

@jit
def pointmass_anomaly_kernel(normx,normy,Lmin,sph_dist):
    kernel = 0
    lik   = np.sqrt(normx**2 + normy**2 - 2*normx*normy*np.cos(sph_dist))
    if Lmin != 0:
        leg = legendre(Lmin,np.cos(sph_dist))
        for i in range(0,Lmin):
            kernel += (1/normy)*((normy/normx)**(i+1))*leg[i]*((i-1)/normx)
    result = (((normx - normy*np.cos(sph_dist))/(lik**3))-(2/(lik*normx))) - kernel
    return(result)

@jit
def pointmass_disturbance_kernel(normx,normy,Lmin,sph_dist):
    kernel = 0
    lik   = np.sqrt(normx**2 + normy**2 - 2*normx*normy*np.cos(sph_dist))
    if Lmin != 0:
        leg = legendre(Lmin,np.cos(sph_dist))
        for i in range(0,Lmin):
            kernel += (1/normy)*((normy/normx)**(i+1))*leg[i]*((i+1)/normx)
    result = (((normx - normy*np.cos(sph_dist))/(lik**3))) - kernel
    return(result)

@jit
def poisson_kernel(normx,normy,Lmin,sph_dist):
    kernel = 0
    lik   = np.sqrt(normx**2 + normy**2 - 2*normx*normy*np.cos(sph_dist))
    if Lmin != 0:
        leg = legendre(Lmin,np.cos(sph_dist))
        for i in range(0,Lmin):
            kernel += (2*i+1)*((normy/normx)**(i+1))*leg[i]
    result = ((normy*(normx**2 - normy**2))/(lik**3)) - kernel
    return(result)

@jit
def poisson_anomaly_kernel(normx,normy,Lmin,sph_dist):
    kernel = 0
    lik   = np.sqrt(normx**2 + normy**2 - 2*normx*normy*np.cos(sph_dist))
    if Lmin != 0:
        leg = legendre(Lmin,np.cos(sph_dist))
        for i in range(0,Lmin):
            kernel += (2*i+1)*((normy/normx)**(i+1))*leg[i]*((i-1)/normx)
    result = (((normy*(normx**3 + np.cos(sph_dist)*normy*normx**2 - 5*normy**2 * normx + 3*np.cos(sph_dist)*normy**3))/(lik**5)) +\
              ((-2*normy*(normx**2 - normy**2))/(normx*lik**3))) - kernel
    return(result)

@jit
def poisson_disturbance_kernel(normx,normy,Lmin,sph_dist):
    kernel = 0
    lik   = np.sqrt(normx**2 + normy**2 - 2*normx*normy*np.cos(sph_dist))
    if Lmin != 0:
        leg = legendre(Lmin,np.cos(sph_dist))
        for i in range(0,Lmin):
            kernel += (2*i+1)*((normy/normx)**(i+1))*leg[i]*((i+1)/normx)
    result = (((normy*(normx**3 + np.cos(sph_dist)*normy*normx**2 - 5*normy**2 * normx + 3*np.cos(sph_dist)*normy**3))/(lik**5))) - kernel
    return(result)

@jit
def poisson_wavelet_kernel(normx,normy,Lmin,sph_dist):
    kernel = 0
    lik   = np.sqrt(normx**2 + normy**2 - 2*normx*normy*np.cos(sph_dist))
    b_0 = 1.0/lik
    b_1 = -(normy-normx*np.cos(sph_dist))/(lik**3)
    b_vector = np.array([b_0,b_1,0.0,0.0,0.0])
    bnm_1 = b_1; bnm_2 = b_0
    beta = np.array([0.0,3.0,17.0,13.0])
    
    for i in range(2,5):
        b_vector[i] = (2*i-1)*lik*b_1*bnm_1 - ((i-1)**2)*(b_0**2)*bnm_2
        bnm_2 = bnm_1
        bnm_1 = b_vector[i]
       
    poisson = 2*(normy**4)*b_vector[4]
    for i in range(1,4):
        poisson += beta[i]*(normy**i)*b_vector[i]
    if Lmin != 0:
        leg = legendre(Lmin,np.cos(sph_dist))
        for i in range(0,Lmin):
            kernel += (i**3)*((normy/normx)**i)*leg[i]*(2*i+1)*(1/normx)
    return(poisson-kernel)
       
@jit
def poisson_wavelet_anomaly_kernel(normx,normy,Lmin,sph_dist):
    kernel = 0
    lik   = np.sqrt(normx**2 + normy**2 - 2*normx*normy*np.cos(sph_dist))
    b_0 = 1.0/lik
    b_1 = -(normy-normx*np.cos(sph_dist))/(lik**3)
    b_vector = np.array([b_0,b_1,0.0,0.0,0.0])
    bnm_1 = b_1; bnm_2 = b_0
    beta = np.array([0.0,3.0,17.0,13.0])
    
    for i in range(2,5):
        b_vector[i] = (2*i-1)*lik*b_1*bnm_1 - ((i-1)**2)*bnm_2*(b_0**2)
        bnm_2 = bnm_1
        bnm_1 = b_vector[i]

    db0 = -(normx-normy*np.cos(sph_dist))/(lik**3)
    db1 = (np.cos(sph_dist)/lik**3) - (3.0/(lik**5))*((np.cos(sph_dist)*normx-normy)*(normx-normy*np.cos(sph_dist)))
    db_vector = np.array([db0,db1,0.0,0.0,0.0])

    b_nm2 = b_0
    b_nm1 = b_1
    db_nm2 = db0
    db_nm1 = db1

    for i in range(2,5):
        term1 = (2*i-1)*((normx-normy*np.cos(sph_dist))/lik)*b_1*b_vector[i-1]
        term2 = (2*i-1)*lik*(db1*b_nm1 + b_1*db_nm1)
        term3 = -((i-1)**2)*(2*b_0*db0*b_nm2 + b_0*b_0*db_nm2)
        db_vector[i] = term1 + term2 + term3
        db_nm2 = db_nm1
        db_nm1 = db_vector[i]
        b_nm2 = b_nm1
        b_nm1 = b_vector[i]
    
    poisson = -2*(normy**4)*((db_vector[4])+((2/normx)*b_vector[4]))
    for i in range(1,4):
        poisson -= beta[i]*(normy**i)*(db_vector[i]+(2/normx)*b_vector[i])
    if Lmin != 0:
        leg = legendre(Lmin,np.cos(sph_dist))
        for i in range(0,Lmin):
            kernel += (i**3)*((normy/normx)**i)*leg[i]*((i-1))*(2*i+1)*(1/(normx**2))
    return(poisson-kernel)

@jit
def poisson_wavelet_disturbance_kernel(normx,normy,Lmin,sph_dist):
    kernel = 0
    lik   = np.sqrt(normx**2 + normy**2 - 2*normx*normy*np.cos(sph_dist))
    b_0 = 1.0/lik
    b_1 = -(normy-normx*np.cos(sph_dist))/(lik**3)
    b_vector = np.array([b_0,b_1,0.0,0.0,0.0])
    bnm_1 = b_1; bnm_2 = b_0
    beta = np.array([0.0,3.0,17.0,13.0])
    
    for i in range(2,5):
        b_vector[i] = (2*i-1)*lik*b_1*bnm_1 - ((i-1)**2)*bnm_2*(b_0**2)
        bnm_2 = bnm_1
        bnm_1 = b_vector[i]

    db0 = -(normx-normy*np.cos(sph_dist))/(lik**3)
    db1 = (np.cos(sph_dist)/lik**3) - (3.0/lik**5)*(np.cos(sph_dist)*normx-normy)*(normx-normy*np.cos(sph_dist))
    db_vector = np.array([db0,db1,0.0,0.0,0.0])

    b_nm2 = b_0
    b_nm1 = b_1
    db_nm2 = db0
    db_nm1 = db1

    for i in range(2,5):
        term1 = (2*i-1)*((normx-normy*np.cos(sph_dist))/lik)*b_1*b_vector[i-1]
        term2 = (2*i-1)*lik*(db1*b_nm1 + b_1*db_nm1)
        term3 = -((i-1)**2)*(2*b_0*db0*b_nm2 + b_0*b_0*db_nm2)
        db_vector[i] = term1 + term2 + term3
        db_nm2 = db_nm1
        db_nm1 = db_vector[i]
        b_nm2 = b_nm1
        b_nm1 = b_vector[i]
    
    poisson = -2*(normy**4)*db_vector[4]
    for i in range(1,4):
        poisson -= beta[i]*(normy**i)*db_vector[i]
    if Lmin != 0:
        leg = legendre(Lmin,np.cos(sph_dist))
        for i in range(0,Lmin):
            kernel += (i**3)*((normy/normx)**i)*leg[i]*((i+1))*(2*i+1)*(1/(normx**2))
    return(poisson-kernel)