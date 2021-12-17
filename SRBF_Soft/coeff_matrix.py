#!/usr/bin/env pypy3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 1 15:26:54 2021

@author: Rasit ULUG
Middle East Technical University

"""

"""
This function constructs design matrix using parallel processing procedure
"""
import numpy as np
import tables
import os
import other_functions

def other_kernel_coeff_mat(rad_dist,lat_sph,lon_sph,Lmin_2,rad_dist_reuter,functions_1,lat_reuter,lon_reuter,workers):
    storage_size = int(len(rad_dist)/10)
    if len(rad_dist)>storage_size:
        indicates = np.append(np.arange(0,len(rad_dist),storage_size),len(rad_dist))
        for i in range(len(indicates)-1):
            sph_dist = other_functions.sph_dist(lat_sph[indicates[i]:indicates[i+1]],lon_sph[indicates[i]:indicates[i+1]],lat_reuter,lon_reuter,workers)
            rad_dist_temp = rad_dist[indicates[i]:indicates[i+1]]
            coeff_mat = other_functions.multiprocessing_model_coeff_pointmass(rad_dist_temp,rad_dist_reuter,Lmin_2,sph_dist,functions_1,other_functions.coeff_mat_pointmass,workers)
            if i == 0:
                f = tables.open_file('coeff_mat.hd5', 'w')
                atom = tables.Atom.from_dtype(coeff_mat.dtype)
                array_c = f.create_earray(f.root, 'data', atom, (0,len(lat_reuter)))
                
                for idx in range(indicates[i+1]):
                    array_c.append(np.array([coeff_mat[idx,:]]))
                f.close()
                sph_dist=None;coeff_mat=None
            else:
                f = tables.open_file('coeff_mat.hd5', mode='a')
                f.root.data.append(coeff_mat)
                f.close()
                sph_dist=None;coeff_mat=None;rad_dist_temp=None  
        f = tables.open_file('coeff_mat.hd5', mode='r')
        coeff_mat = f.root.data[:]
        f.close()
        os.remove('coeff_mat.hd5')
    else:
        sph_dist = other_functions.sph_dist(lat_sph,lon_sph,lat_reuter,lon_reuter,workers) 
        coeff_mat = other_functions.multiprocessing_model_coeff_pointmass(rad_dist,rad_dist_reuter,Lmin_2,sph_dist,functions_1,other_functions.coeff_mat_pointmass,workers)
        sph_dist = None
    return(coeff_mat)
