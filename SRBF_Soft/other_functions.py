#!/usr/bin/env pypy3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:11:21 2018

@author: Rasit ULUG
Middle East Technical University
"""


import numpy as np
from numba import jit
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys

"""
This file contains some necessary functions.They are:
    Spherical distance
    Coordinate Transformation
    Coefficient Matric function
    Some file reading
    Multi-processing function
"""
@jit    
def sph_func(args):
    lan1 = args[0];lon1 = args[1];lan2 = args[2];lon2=args[3]
    lan1rd = lan1 * np.pi/180
    lon1rd = lon1 * np.pi/180
    
    lan2rd = lan2 * np.pi/180
    lon2rd = lon2 * np.pi/180
        
    m,n = len(lan1rd),len(lan2rd)
    sp_dist = np.zeros((m,n))
    
    for k in range(n):
        sp_dist[:,k] = np.arccos(np.clip(np.sin(lan1rd)*np.sin(lan2rd[k]) + np.cos(lan1rd)*np.cos(lan2rd[k])*np.cos(lon1rd-lon2rd[k]),-1,1))
    return(sp_dist)
    
def sph_dist(lan1,lon1,lan2,lon2,workers,sph_func=sph_func):
    if len(lan1)>workers*1000:
        length = workers*1000
    elif 1000<=len(lan1)<=workers*1000:
        length = 1000
    else:
        coeff_mat = sph_func([lan1,lon1,lan2,lon2])
        return(coeff_mat)
    scale = int(np.floor(length/workers))
    
    size = int(np.floor(len(lan1)/length))
    indis = np.arange(0,len(lan1),scale)
    indis = np.append(indis,len(lan1))
    
    for i in range(size+1):
        if i<=(size-1):
            args = [None]*workers
            for k in range(workers):
                args[k] = [lan1[indis[workers*i+k]:indis[workers*i+1+k]],lon1[indis[workers*i+k]:indis[workers*i+1+k]],lan2,lon2]
                
        else:
            indis_new = np.linspace(indis[workers*i],indis[-1],num=workers,endpoint=True,dtype=int)
            args = [None]*(workers-1)
            new_i = 0
            for k in range(workers-1):
                args[k] =  [lan1[indis_new[workers*new_i+k]:indis_new[workers*new_i+1+k]],lon1[indis_new[workers*new_i+k]:indis_new[workers*new_i+1+k]],lan2,lon2]
        pool = Pool()
        results = pool.map_async(sph_func,args,1).get()
        if i == 0:
            coeff_mat = np.concatenate(results)
        else:
            coeff_mat = np.concatenate((coeff_mat,np.concatenate(results)))
        del results
        pool.close()
        pool.terminate()
        pool.join()
    return(coeff_mat)  

def geodetic2cartesian(fi,lamda,h):
    a = 6378137.0
    e2 = 0.00669438002290
    
    fi = fi*np.pi/180
    lamda = lamda*np.pi/180
    
    N = a/(np.sqrt(1-e2*np.sin(fi)**2))
    
    x = np.cos(fi)*np.cos(lamda)*(N+h)
    y = np.cos(fi)*np.sin(lamda)*(N+h)
    z = np.sin(fi)*((1-e2)*N+h)
    r = np.transpose(np.array((x,y,z)))
    return(r)
    
def car2sph(x,y,R):
    
    lamda = np.arctan2(y,x)*180/np.pi
    
    fi = np.arccos(np.sqrt(x**2 + y**2)/R)*180/np.pi
    
    return(fi,lamda)

def sph2car(lat,lon,R):
    lat = lat*np.pi/180.0
    lon = lon*np.pi/180.0
    
    x = R*np.cos(lat)*np.cos(lon)
    y = R*np.cos(lat)*np.sin(lon)
    
    return(x,y)

def ell2sph(fi,lamda,h):
    a = 6378137.0
    e2 = 0.00669438002290
    
    fi = fi*np.pi/180
    
    N = a/(np.sqrt(1-e2*np.sin(fi)**2))
    
    fi = np.arctan(((N+h)*np.cos(fi))/(((N*(1-e2)+h))*np.sin(fi)))
    r = np.sqrt((((N+h)*np.cos(fi))**2)+(((N*(1-e2)+h))*np.sin(fi))**2)
    for i in range(len(fi)):
        if fi[i]*180/np.pi<0:
            fi[i] = 90+(fi[i]*180/np.pi)
        else:
            fi[i] = 90-(fi[i]*180/np.pi)     
    return(fi,lamda,r)    
    
@jit
def coeff_mat_pointmass(args):
    locdata=args[0];locdata_grid=args[1];Lmin=args[2];spdist=args[3];functions=args[4]
    coeff_mat = np.zeros((np.shape(spdist)[0],np.shape(spdist)[1]))
    for i in range(np.shape(spdist)[0]):
        for k in range(np.shape(spdist)[1]):
            coeff_mat[i,k] = functions(locdata[i],locdata_grid[k],Lmin,spdist[i,k])
    return(coeff_mat)

    
def multiprocessing_model_ggm(C1,S1,Lmax,locdata,GM1,radius1,kind,functions,workers):
    
    if len(locdata)>workers*1000:
        length = workers*1000
    elif 1000<=len(locdata)<=workers*1000:
        length = 1000
    else:
        coeff_mat = functions([C1,S1,Lmax,locdata,GM1,radius1,kind])
        return(coeff_mat)
    scale = int(np.floor(length/workers))
    
    size = int(np.floor(len(locdata)/length))
    indis = np.arange(0,len(locdata),scale)
    indis = np.append(indis,len(locdata))
    
    for i in range(size+1):
        if i<=(size-1):
            args = [None]*workers
            for k in range(workers):
                args[k] = [C1,S1,Lmax,locdata[indis[workers*i+k]:indis[workers*i+1+k]],GM1,radius1,kind]
        else:
            indis_new = np.linspace(indis[workers*i],indis[-1],num=workers,endpoint=True,dtype=int)
            args = [None]*(workers-1)
            new_i = 0
            for k in range(workers-1):
                args[k] = [C1,S1,Lmax,locdata[indis_new[workers*new_i+k]:indis_new[workers*new_i+1+k]],GM1,radius1,kind]
            
        pool = Pool()
        results = pool.map_async(functions,args,1).get()  
        
        if i == 0:
            coeff_mat = np.concatenate((results[:]))
        else:
            coeff_mat = np.concatenate((coeff_mat,np.concatenate((results[:]))))
            
        pool.close()
        pool.terminate()
        pool.join()

    return(coeff_mat)

def delta_C(C,GM,R,GMEl=3986005*(10**8),radius=0.6378137E+07):
#   R and GM belong GGM, radius and GMEl belong GRS80 ellipsoid 
    C_2 = np.zeros((np.shape(C)[0],np.shape(C)[1]))
    J2 = (-108263*(10**-8))/np.sqrt(5)
    e = np.sqrt(0.006694380022903416)
    for n in range(11):
        C_2[n*2,0] = (((-1)**n)*((3*e**(2*n)))/((2*n+1)*(2*n+3)*np.sqrt(4*n+1)))*(1-n-(5**(3/2))*n*J2/(e**2))*((radius/R)**(2*n))*(GMEl/GM)
    return(C-C_2)


def multiprocessing_model_coeff_pointmass(locdata,locdata_grid,Lmin,sph_dist,functions,functions_1,workers):
    
    if len(locdata)>workers*1000:
        length = workers*1000
    elif 1000<=len(locdata)<=workers*1000:
        length = 1000
    else:
        coeff_mat = functions_1([locdata,locdata_grid,Lmin,sph_dist,functions])
        return(coeff_mat)
    
    scale = int(np.floor(length/workers))
    
    size = int(np.floor(len(locdata)/length))
    indis = np.arange(0,len(locdata),scale)
    indis = np.append(indis,len(locdata))
    
    for i in range(size+1):
        if i<=(size-1):
            args = [None]*workers
            for k in range(workers):
                args[k] = [locdata[indis[workers*i+k]:indis[workers*i+1+k]],locdata_grid,Lmin,sph_dist[indis[workers*i+k]:indis[workers*i+1+k]],functions]

        else:
            indis_new = np.linspace(indis[workers*i],indis[-1],num=workers,endpoint=True,dtype=int)
            args = [None]*(workers-1)
            new_i = 0
            for k in range(workers-1):
                args[k] = [locdata[indis_new[workers*new_i+k]:indis_new[workers*new_i+1+k]],locdata_grid,Lmin,sph_dist[indis_new[workers*new_i+k]:indis_new[workers*new_i+1+k]],functions]
   
        pool = Pool()

        results = pool.map_async(functions_1,args,1).get()

        if i == 0:
            coeff_mat = np.concatenate(results)
        else:
            coeff_mat = np.concatenate((coeff_mat,np.concatenate(results)))

        pool.close()
        pool.terminate()
        pool.join()

        del results
    return(coeff_mat)

def read_file(filename):
    file = open(filename,encoding="utf8", errors='ignore')
    lines = file.read().splitlines()
    length = len(lines)
    lat  = np.zeros(length)
    lon  = np.zeros(length)
    H    = np.zeros(length)
    grav = np.zeros(length)
    for i in range(length):
        line = lines[i].split()
        lat[i] = np.float(line[1])
        lon[i] = np.float(line[2])
        H[i]   = np.float(line[3])
        grav[i]= np.float(line[4])
    return(lat,lon,H,grav)

def read_gnss_file(filename):
    file = open(filename,encoding="utf8", errors='ignore')
    lines = file.read().splitlines()
    length = len(lines)
    lat  = np.zeros(length)
    lon  = np.zeros(length)
    H    = np.zeros(length)
    for i in range(length):
        line = lines[i].split()
        lat[i] = np.float(line[1])
        lon[i] = np.float(line[2])
        H[i]   = np.float(line[3])
    return(lat,lon,H)
   
def MCVCE(coeff_mat,grav):
    data_number = len(coeff_mat)
    normal_matrix = [None]*data_number
    obs_vector = [None]*data_number
    r = np.zeros(data_number+1)
    sigma = [np.array([1])]*(data_number+1)
    sigma_dif = np.zeros(data_number+1)
    u2 = [None]*data_number
    for i in range(data_number):
        normal_matrix[i] = (coeff_mat[i].T @ coeff_mat[i])
        obs_vector[i] = np.dot(coeff_mat[i].T,grav[i])
        
    m,n = np.shape(normal_matrix[0])
    Pu = np.eye(n)
    for i in range(data_number):
        u2[i] = np.random.binomial(1,0.5,len(coeff_mat[i]))
        u2[i][u2[i]==0] = -1
    u = np.random.binomial(1,0.5,n)
    u[u==0] = -1
    while True:
        Qd = np.zeros((m,n))
        Qn = np.zeros(m)
    
        for i in range(data_number):
            Qd += (1/(sigma[i][-1]**2))*normal_matrix[i] 
            Qn += (1/sigma[i][-1]**2)*obs_vector[i]
        Qd += (1/(sigma[-1][-1]**2))*Pu
        
        parameter = np.linalg.solve(Qd,Qn)
                
        for i in range(data_number):
            Beta = np.linalg.solve(Qd,np.dot(coeff_mat[i].T,u2[i]))
            r[i] = len(coeff_mat[i])-(1/sigma[i][-1]**2)*np.dot(u2[i].T,np.dot(coeff_mat[i],Beta))
            
            sigma[i] = np.append(sigma[i],np.sqrt(np.sum((np.dot(coeff_mat[i],parameter)-grav[i])**2)/r[i]))
                
        r[-1] = n - (1/(sigma[-1][-1]**2))*np.dot(u.T,np.linalg.solve(Qd,u))
        sigma[-1] = np.append(sigma[-1],np.sqrt(np.sum(parameter**2)/r[-1]))
        
        for i in range(data_number+1):
            sigma_dif[i] = sigma[i][-1] - sigma[i][-2]               
        if np.all(np.abs(sigma_dif)<0.01):
            return(parameter,sigma)

    return(parameter,sigma)

def LS(coeff_mat,grav):
    data_number = len(coeff_mat)
    normal_matrix = [None]*data_number
    obs_vector = [None]*data_number
    
    for i in range(data_number):
        normal_matrix[i] = (coeff_mat[i].T @ coeff_mat[i])
        obs_vector[i] = np.dot(coeff_mat[i].T,grav[i])
        
    m,n = np.shape(normal_matrix[0])
    Qd = np.zeros((m,n))
    Qn = np.zeros(m)

    for i in range(data_number):
        Qd += normal_matrix[i] 
        Qn += obs_vector[i]

    parameter = np.linalg.solve(Qd,Qn)
    
    return(parameter)

def test_normal_eqution(coeff_mat):
    data_number = len(coeff_mat)
    normal_matrix = [None]*data_number
    
    for i in range(data_number):
        normal_matrix[i] = (coeff_mat[i].T @ coeff_mat[i])
    
    m,n = np.shape(normal_matrix[0])
    Qd = np.zeros((m,n))
    for i in range(data_number):
        Qd += normal_matrix[i]
    
    if np.linalg.cond(Qd) < 1/sys.float_info.epsilon:
        regu = 0
    else:
        regu = 1
    
    return(regu)


def sel_turn_frm_graph(Lmin_range,rms_control):
    print('Please select Lmin value from graph...')
    plt.plot(Lmin_range,rms_control)
    plt.ylim(np.floor(np.min(rms_control)),np.max(rms_control))
    Lmin_last = plt.ginput(1,timeout=0)
    plt.close()
    Lmin_last = Lmin_range[np.argmin(np.abs(Lmin_range - Lmin_last[0][0]))]
    Lmin_indis = np.where(Lmin_last == Lmin_range)[0][0]
    return(Lmin_last,Lmin_indis)
    

def read_ggm(file,n):
    rnxfile = open(file,encoding="utf8", errors='ignore')
    lines = rnxfile.read().splitlines()
    n += 1
    
    i = 0
    while lines[i].find('end_of_head') == -1:
        i += 1

    C = np.zeros((n,n));S = np.zeros((n,n))
    indis = 0
    for deg in range(n):
        for order in range(deg+1):
            i += 1
            if lines[i].split()[0] != 'gfc':  i+=1
                
            if indis != int(lines[i].split()[1]):
                C[deg][order] = 0
                S[deg][order] = 0
                i -=1
                indis = deg
            else:
                try:
                    line = lines[i].split()
                    C[deg][order] = np.float(line[3])
                    S[deg][order] = np.float(line[4])
                except:
                    line = lines[i].split()
                    C[deg][order] = np.float(line[3].upper().replace('D','E'))
                    S[deg][order] = np.float(line[4].upper().replace('D','E'))
        indis += 1
    try:
        i = 0
        while lines[i].find('earth_gravity_constant') == -1:
            i += 1
        try:
            GM = np.float(lines[i].split()[1])
        except:
            GM = np.float(lines[i].split()[1].upper().replace('D','E'))
            
        i = 0
        while lines[i].find('radius') == -1:
            i += 1
        try:    
            radius = np.float(lines[i].split()[1])
        except:
            radius = np.float(lines[i].split()[1].upper().replace('D','E'))
    except:
        GM = 0
        radius = 0
    return(C,S,GM,radius)
