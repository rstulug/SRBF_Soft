#!/usr/bin/env pypy3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:13:16 2020

@author: rst
"""
from numba import jit
import numpy as np
import other_functions
import coeff_matrix
from sklearn import cluster

@jit
def k_SRBF(lat,lon,k_init,min_p,min_d,max_d,R,workers):
    lat_all = []
    lon_all = []

    for i in range(len(lat)):
        lat_all = np.append(lat_all,lat[i])
        lon_all = np.append(lon_all,lon[i])
    
    car_x,car_y = other_functions.sph2car(lat_all,lon_all,R)
    data_all = np.empty((len(lat_all),2))
    data_all[:,0] = car_x
    data_all[:,1] = car_y
    kmeans = cluster.KMeans(n_clusters=k_init,n_jobs=-1,n_init=1,max_iter=1000,tol=1e-10).fit(data_all)
    
    data_all = None
    C_clustered = kmeans.cluster_centers_
    carx_rbf = C_clustered[:,0]
    cary_rbf = C_clustered[:,1]
    lat_rbf,lon_rbf = other_functions.car2sph(carx_rbf,cary_rbf,R)
    idx = kmeans.labels_
##############################################################################    
    m = len(lat_rbf)

    total = np.zeros(m)
    for i in range(m):
        total[i] = len(np.where(idx == i)[0])
        
    index_minp = np.where(total<min_p)[0]
    
    if len(index_minp)>0:
        for i in index_minp:
            
            sph_del = other_functions.sph_dist(np.array([lat_rbf[i]]),np.array([lon_rbf[i]]),lat_rbf,lon_rbf,workers)
            sec_small = np.argsort(sph_del[0,:])[1]
            
            idx[np.where(idx == i)[0]] = sec_small
            
            carx_rbf[sec_small]  = np.mean(car_x[np.where(idx==sec_small)[0]])
            cary_rbf[sec_small]  = np.mean(car_y[np.where(idx==sec_small)[0]])
            lat_rbf[sec_small],lon_rbf[sec_small] = other_functions.car2sph(carx_rbf[sec_small],cary_rbf[sec_small],R)
            
            lat_rbf = np.delete(lat_rbf,i)
            lon_rbf = np.delete(lon_rbf,i)
            
            carx_rbf = np.delete(carx_rbf,i)
            cary_rbf = np.delete(cary_rbf,i)
            
            idx[np.where(idx>i)[0]] -= 1
            index_minp[np.where(index_minp>i)[0]] -= 1
        else:
            pass
##############################################################################    
    sph_merge = other_functions.sph_dist(lat_rbf, lon_rbf, lat_rbf, lon_rbf, workers)*180/np.pi
    np.fill_diagonal(sph_merge,10)
    sayac = 1
    while sayac == 1:
        indis = np.where(sph_merge<min_d)
        if len(indis[0])>0:
            
            carx_rbf[indis[0][0]] = np.mean(np.append(car_x[np.where(idx == indis[0][0])[0]],car_x[np.where(idx == indis[1][0])[0]]))
            cary_rbf[indis[0][0]] = np.mean(np.append(car_y[np.where(idx == indis[0][0])[0]],car_y[np.where(idx == indis[1][0])[0]]))
            lat_rbf[indis[0][0]],lon_rbf[indis[0][0]] = other_functions.car2sph(carx_rbf[indis[0][0]], cary_rbf[indis[0][0]], R)
            
            lat_rbf = np.delete(lat_rbf,indis[1][0])
            lon_rbf = np.delete(lon_rbf,indis[1][0])
            
            carx_rbf = np.delete(carx_rbf,indis[1][0])
            cary_rbf = np.delete(cary_rbf,indis[1][0])
            
            idx[np.where(idx==indis[1][0])[0]] = indis[0][0]
            idx[np.where(idx>indis[1][0])[0]] -= 1
            
            sph_merge = np.delete(sph_merge,indis[1][0],0)
            sph_merge = np.delete(sph_merge,indis[1][0],1)
            
            sph_new = other_functions.sph_dist(np.array([lat_rbf[indis[0][0]]]),np.array([lon_rbf[indis[0][0]]]),lat_rbf,lon_rbf,12)*180/np.pi
            sph_new[0,indis[0][0]] = 10
            sph_merge[indis[0][0],:] = sph_new
            sph_merge[:,indis[0][0]] = sph_new            
        else:
            sayac = 0
##############################################################################  
    next_s = 1
    m = len(lat_rbf)
    total_ind = [None]*m
    for i in range(m):
        total_ind[i] = np.where(idx== i)[0]
    
    indis1 = 0  
    while next_s == 1:
        for i in range(indis1,m):
            sph_splt = other_functions.sph_dist(np.array([lat_rbf[i]]),np.array([lon_rbf[i]]),lat_all[total_ind[i]],lon_all[total_ind[i]],workers)*180/np.pi
            if len(np.where(sph_splt>=max_d)[1])>=1 and len(sph_splt[0,:])>=2*min_p:
                indis1 = i
                break
            else:
                indis1 = i
            
        if indis1+1 == m:
            next_s = 0
        else:       
            data = np.empty((len(lat_all[total_ind[indis1]]),2))
            data[:,0] = car_x[total_ind[indis1]]
            data[:,1] = car_y[total_ind[indis1]]
            kmeans = cluster.KMeans(n_clusters=2,n_init=1,max_iter=500,tol=1e-10,n_jobs=-1).fit(data)
            C_clustered1 = kmeans.cluster_centers_
            
            carx_rbf1 = C_clustered1[:,0]
            cary_rbf1 = C_clustered1[:,1]
            lat_rbf1,lon_rbf1 = other_functions.car2sph(carx_rbf1,cary_rbf1,R)
            idx_clustred1 = kmeans.labels_
            sph_test = other_functions.sph_dist(lat_rbf1,lon_rbf1,np.delete(lat_rbf,indis1),np.delete(lon_rbf,indis1),workers)*180/np.pi
            sph_test2 = other_functions.sph_dist(np.array([lat_rbf1[0]]),np.array([lon_rbf1[0]]),np.array([lat_rbf1[1]]),np.array([lon_rbf1[1]]),workers)*180/np.pi
           
            if np.all(sph_test>min_d) and np.all(sph_test2>min_d) and len(np.where(idx_clustred1==0)[0])>min_p and len(np.where(idx_clustred1==1)[0])>min_p:
                lat_rbf = np.append(lat_rbf,lat_rbf1)
                lon_rbf = np.append(lon_rbf,lon_rbf1)
                
                carx_rbf = np.append(carx_rbf,carx_rbf1)
                cary_rbf = np.append(cary_rbf,cary_rbf1)
                
                total_ind.append(total_ind[indis1][np.where(idx_clustred1 == 0)[0]])
                total_ind.append(total_ind[indis1][np.where(idx_clustred1 == 1)[0]])
                
                lat_rbf = np.delete(lat_rbf,indis1)
                lon_rbf = np.delete(lon_rbf,indis1)
                carx_rbf = np.delete(carx_rbf,indis1)
                cary_rbf = np.delete(cary_rbf,indis1)
                total_ind.pop(indis1)
                m += 1
            else:
                indis1 += 1
##############################################################################  
    return(lat_rbf,lon_rbf)
        
def depth_sel_with_GCV(lat_sph,lon_sph,rad_dist,lat_srbf,lon_srbf,residual_data,R_bjerhammer,function,workers,depth_range,regu,norm_fac,total_data):
    
    print('\nOptimal depth selection started...')
    k = 0
    data_number = len(lat_sph)
    coeff_mat = [None]*data_number
    gcv = np.zeros(len(depth_range))
    u = np.random.binomial(1,0.5,total_data)
    u[u==0] = -1
    for j in depth_range:
        print('\nTesting for {:.1f} depth. Please wait...'.format(j))
        rad_dist_rbf = np.ones(len(lat_srbf))*(R_bjerhammer-j)*1e3*norm_fac
        
        for i in range(data_number):
            coeff_mat[i] = coeff_matrix.other_kernel_coeff_mat(rad_dist[i],lat_sph[i],lon_sph[i],0,rad_dist_rbf,function[i],lat_srbf,lon_srbf,workers)
        print('\nParameter Estimation started. Please wait...')
        if regu == 0:
            parameter = other_functions.LS(coeff_mat,residual_data)
        else: 
            parameter,sigma = other_functions.MCVCE(coeff_mat,residual_data)
            
        all_coeff = np.concatenate((coeff_mat[:]))
        coeff_mat = [None]*data_number
        all_res = np.concatenate((residual_data[:]))
        
        if regu == 0:
            Beta = np.linalg.solve(np.dot(all_coeff.T,all_coeff),np.dot(all_coeff.T,u))
            trace = np.dot(u.T,np.dot(all_coeff,Beta))
            gcv[k] = (total_data*np.sum(((np.dot(all_coeff,parameter)-all_res)*1e-5)**2)) / ((total_data-trace)**2)
        else:
            all_coeff2 = np.copy(all_coeff)
            weight = np.ones(total_data)
            indis = 0
            for i in range(data_number):
                all_coeff2[indis:indis+len(lat_sph[i]),:] *= 1/(sigma[i][-1]**2)
                weight[indis:indis+len(lat_sph[i])] *= 1/(sigma[i][-1]**2)
                indis += len(lat_sph[i])
            Beta = np.linalg.solve((np.dot(all_coeff2.T,all_coeff)+(sigma[0][-1]**2/sigma[-1][-1]**2)*np.eye(len(parameter))),np.dot(all_coeff2.T,u))
            trace = np.dot(u.T,np.dot(all_coeff,Beta))
            gcv[k] = (total_data*np.sum((((np.dot(all_coeff,parameter)-all_res)*1e-5)**2)*weight)) / ((total_data-trace)**2)
            all_coeff = None
            all_coeff2 = None
            weight = None
        print('{:.1f} km depth is tested, GCV is {:.6e}'.format(j,gcv[k]))
        k += 1
    
    depth_last = depth_range[np.argmin(gcv)]
    return(depth_last,gcv)
        
