#!/usr/bin/env pypy3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 1 15:26:54 2021

@author: Rasit_ULUG
Middle East Technical University
"""
# Importing modules
import os
import sys
import time
import multiprocessing
import numpy as np
import warnings
import gc
import coeff_matrix
import kernel_functions
import other_functions
import gravity_functions
import RTM_corrections
import k_SRBF

warnings.filterwarnings('ignore')
nw_first = time.time()
##############################################################################
print("\nSRBF_Soft: The regional gravity field modeling software via spherical radial basis functions\n")
print('\nReference ellipsoid: GRS80')
##############################################################################
print('\nIn this version, you can only use gravity anomaly and disturbance as input and control data')
res_sel = 0
print('\nIf you use residual gravity anomalies/distrubances, please enter 1 otherwise enter 0')
res_sel = int(input(': '))

if res_sel in (0,1):
    pass
else:
    raise ValueError("Wrong format chosen")
##############################################################################
data_number = int(input('\nPlease enter how many different type of data you will use for modeling: '))

#Empty list definition to store data
lat = [None]*data_number
lon = [None]*data_number
H =[None]*data_number
grav = [None]*data_number
residual_data = [None]*data_number
data_type = [None]*data_number
total_data = 0
#Importing data
for i in range(data_number):
    
    print('\nPlease import {}. data file: '.format(i+1))
    print('\nData file format: "Point Id" "Latitude" "Longitude" "Height (meter)" "Data(Anomaly or Disturbance (mGal))"')
    data_file = input('{}. Data File: '.format(i+1))

    try:
        if res_sel == 1:
            lat[i],lon[i],H[i],residual_data[i] = other_functions.read_file(data_file)
        else:
            lat[i],lon[i],H[i],grav[i] = other_functions.read_file(data_file)
        total_data += len(lat[i])
        if min(lon[i]) < -180 or max(lon[i])> 180 or min(lat[i])<-90 or max(lat[i])>90:
            raise ValueError('Latitude and Longitude values are wrong (Lat: -90 to 90,  Lon: -180 to 180')
    except:
        raise AttributeError("File does not exist or file format is wrong!")
    
    print('\nPlease Enter the type of {}. data: '.format(i+1))
    print('\n1: Gravity Anomaly \n2: Gravity Disturbance ')
    
    data_type[i] = int(input('\ntype of {}. data: '.format(i+1)))
    

##############################################################################
control_data = 2
print('\nDo you want to use external control points to validate results')
print('\nNote: If you want to use turning point algorithm, you have to give external control points')
control_data = int(input('\n1: Yes \n2: No  \n: '))

if control_data not in (1,2):
    raise ValueError('You choose wrong control data option. Enter 1 to use external control data, 2 for No')

if control_data == 1:
    print('\nPlease import control data file: ')
    print('\nData file format: "Point Id" "Latitude" "Longitude" "Height (meter)" "Data(Anomaly or Disturbance (mGal))"')
    control_file = input('\nControl Points File: ')
    try:
        if res_sel == 1:
            lat_control,lon_control,H_control,residual_data_control = other_functions.read_file(control_file)
        else:
            lat_control,lon_control,H_control,grav_control = other_functions.read_file(control_file)
    except:
        raise AttributeError("File does not exist or file format is wrong!")  
    
    print('\nPlease enter type of control data')
    print('\n1: Gravity Anomaly \n2: Gravity Disturbance')
    control_data_type = int(input('\ntype of control data: '))

##############################################################################
print('\nDo you want to compute Height-Anomaly ? ')
qgeoid = int(input('\n1: Yes \n2: No  \n: '))
if qgeoid == 1:
    print('\nFile format: "Point Id"  "Latitude" "Longitude" "Height"')    
    qgeoid_file = input('Filename: ')
    try:
        qgeoid_lat,qgeoid_lon,qgeoid_H = other_functions.read_gnss_file(qgeoid_file)
    except:
        raise AttributeError("File does not exist or file format is wrong!") 
##############################################################################
print('\nDefine Parameters')

depth_choose = 1
Lmin_choose = 1

if res_sel == 0:
    print('\nPlease enter model parameters correctyly')
    Lmin =  int(input('\nEnter degree of Global Geopotential Model (GGM) for long-wavelength computation\n: '))
    if Lmin<0: sys.exit('Lmin can not be lower than 0')
    
    print('\nImport Global Geopotential Model: ')
    print('\nLong wavelength of gravity signal computed from GGM model up to {} degree and order will be extracted'.format(Lmin))
    
    back_ggm = input('\nGGM File\n: ')
    if os.path.isfile(back_ggm) == False:
        sys.exit('{} file does not found '.format(back_ggm))
    
    # Reading GGM model coefficients and subtract reference ellipsoid coefficients
    try:
        # GGM coefficients for Remove part of RCR 
        C,S,GM,radius = other_functions.read_ggm(back_ggm,Lmin)
        if GM == 0 or radius == 0:
            print('Could not read radius and earth_gravity_constant from GGM.Please input manually:')
            radius   = float(input('\nRadius of GGM: '))
            GM       = float(input('\nEarth Gravity Constant of GGM: '))
        C = other_functions.delta_C(C,GM,radius)
    except:
        raise Exception("\nWrong GGM (or GGM format) selected")
##############################################################################
if res_sel == 0:
    print("\nSRBF_Soft uses Gravsoft TC module as subprocess to calculate Residual Terrain Effects ")
    print('\nFor parameters please look Gravsoft User Manuel (Forsberg, R., & Tscherning, C. C. (2008). An overview manual for the GRAVSOFT geodetic gravity field modelling programs. Contract report for JUPEM.)')
    print("\nPut DEM files in DEM folder for uniform directory lookup ")
    print('Due to all DEM file must be putted in DEM directory, please enter just DEM name only')
    
    print('\nImport Detailed DEM file for Gravsoft TC module: ')
    dem_file = input('\nDetailed DEM File: ')
    if os.path.isfile('DEM/'+dem_file) == False:
        raise Exception("{} file does not found ".format(dem_file))
    
    print('\nImport coarse DEM file for Gravsoft TC module (could be created using Gravsoft SELECT module): ')
    coarse_file = input('\nCoarse DEM File: ')
    if os.path.isfile('DEM/'+coarse_file) == False:
        raise Exception("{} file does not found ".format(coarse_file))
    
    print('\nImport reference DEM file for Gravsoft TC module(could be created using Gravsoft TCGRID module): ')    
    ref_file = input('\nReference DEM File: ')
    if os.path.isfile('DEM/'+ref_file) == False:
        raise Exception("{} file does not found ".format(ref_file))
    
    izcode = [None]*data_number
    density = [None]*data_number
    
    density = float(input('Enter density (Suggested For Land 2.67, for water 1.64): '))
    
    print('\nEnter izcode for {}.th data \n\n0: station on terrain, change station elevation \n1: do,change terrain model \n2: do,change terrain model on land points only'.format(i+1))
    print('3: station free \n4: do, station free, no spline densification')  
    for i in range(data_number):
    
        izcode[i] = int(input('izcode for {}.th data: '.format(i+1)))
        if izcode[i] not in (0,1,2,3,4):
            raise ValueError('Wrong izcode selected') 
        
    if control_data == 1:
        print('\nPlease ener izcode for external control points')
        izcode_control = int(input('izcode for control points: '))
       
    r1,r2 = input('\nEnter inner and outer computation radius respectively (km) (e.g. 10 200): ').split()
    try:
        r1 = float(r1)
        r2 = float(r2)
    except:
        raise ValueError("Inner and outer radius are not float or interger")
        
    print('\nEnter Maximum Area To Computation for Gravsoft TC module')
    lat1_dem,lat2_dem,lon1_dem,lon2_dem = input(' "Lowest Lat." "Highest Lat." "Lowest Lon." "Highest Lon" (e.g. 42 50 -2 8)\n: ').split()
    
    try:
        lat1_dem = float(lat1_dem)
        lat2_dem = float(lat2_dem)
        lon1_dem = float(lon1_dem)
        lon2_dem = float(lon2_dem)
    except:
        raise ValueError("Area borders are not float  or integer")   
##############################################################################
kernel_type = int(input('\nPlease Select Kernel Type:\n1:Point Mass Kernel \n2:Poisson Kernel \n3:Poisson wavelet (order 3) Kernel\n: '))

if kernel_type in (1,2,3):
    pass
else:
    raise ValueError("Wrong kernel type chosen")
##############################################################################
print('\nDefine parameters for k-SRBF algorihtm ')

k_init = int(input('\nPlease enter the k number for the construction of the initial candidate network: '))
k_min  = int(input('\nPlease enter the minimum sample number for each cluster: ')) 
min_srbf = float(input('\nPlease enter the minimum spherical distance between the centroids (degree)): '))
max_srbf = float(input('\nPlease enter the maximum spherical distance between centroid and its samples to split the cluster (degree)): '))
##############################################################################
depth_choose = int(input('\nDo you wanna enter bandwidth (depth) directly (enter 1) or choosing best one in a specified range (enter 2): '))   


if depth_choose == 2:
    depth_i,depth_l,depth_s = input('\nEnter bandwidth (depth) range (km) (Initial Last Step Size (e.g. 5 20 1)): ').split()
    try:
        depth_i = float(depth_i)
        depth_l = float(depth_l)
        depth_s = float(depth_s)
    except:
        raise ValueError("Please enter depth correctly. Initial Values End Values Step Size (e.g 5 20 1)")   
else:
    depth = float(input('\nEnter depth of SRBFs center (km): '))
##############################################################################
if qgeoid == 1:    
    print('\nHere we follow turning point algorithm (see doi: 10.1016/j.jog.2019.01.001)')
    print('\nUser will choose turning point from graph')
    Lmin_choose = int(input('\nDo you wanna enter reduced SRBF value directly (enter 1) or choosing from graph (turning point) (enter 2): '))

    if Lmin_choose == 2 and control_data !=1:
        raise Exception("Without control data, you cannot select a reduced SRBF range")

    if Lmin_choose == 2:
        Lmin_i,Lmin_l,Lmin_s = input('\nPlease choose reduced SRBFs parameter correctly "Initial values" "End Values" "Step Size" (e.g. 21 361 20): ').split()
        try:
            Lmin_i = int(Lmin_i)
            Lmin_l = int(Lmin_l)
            Lmin_s = int(Lmin_s)
        except:
            raise ValueError('Wrong range selected. (e.g. 21 361 20)')
    else:
        try:
            Lmin_2 = int(input('\nEnter reduced SRBF parameter (Default is 0): '))
        except:
            Lmin_2 = 0
else:            
    Lmin_2 = 0
            
try:
    R_bjerhammer = float(input('\nEnter Radius of Bjerhammer Sphere (km) (Default is 6371.0): '))
except:
    R_bjerhammer = 6371.0

try:
    workers = int(input('\nEnter number of CPU to use in process. It enables parallelization and decrease the computation time. (Default is Total_CPU-1) : ' ))
except:
    workers = multiprocessing.cpu_count()-1
    
gc.collect() 
##############################################################################
print('\n\nHere we go...')
##############################################################################
if res_sel == 0:
    # Terrain Correction Calculation (Using Gravsoft TC module with subprocess library in python)
    print('\nResidual Terrain Model (RTM) calculation started...')
    nw = time.time()
    
    rtm = [None]*data_number
    
    for i in range(data_number):
        if data_type[i] == 1:
            itype = 5
        else:
            itype = 1
        data_tc = np.transpose(np.array([lat[i],lon[i],H[i],grav[i]]))
        terr_cor = RTM_corrections.terrain_correction_multi(data_tc,dem_file,coarse_file,ref_file,lat1_dem,lat2_dem,lon1_dem,lon2_dem,itype,4,izcode[i],1,density,r1,r2,workers)
        data_tc = None
        
        ide,rtm[i] = RTM_corrections.read_rtm(terr_cor)
        ide = ide.astype(int)
        
        lat[i] = lat[i][ide]
        lon[i] = lon[i][ide]
        H[i]   = H[i][ide]
        grav[i] = grav[i][ide]
    
    if control_data == 1:
        if control_data_type == 1:
            itype = 5
        else:
            itype = 1
    
        data_tc = np.transpose(np.array([lat_control,lon_control,H_control,grav_control]))
        terr_cor = RTM_corrections.terrain_correction_multi(data_tc,dem_file,coarse_file,ref_file,lat1_dem,lat2_dem,lon1_dem,lon2_dem,itype,4,izcode_control,1,density,r1,r2,workers)
    
        data_tc = None
        ide,rtm_control = RTM_corrections.read_rtm(terr_cor)
        ide = ide.astype(int)
        
        lat_control = lat_control[ide]
        lon_control = lon_control[ide]
        H_control   = H_control[ide]
        grav_control = grav_control[ide]
    gc.collect()
    print('Terrain Correction Calculation completed.Process time:{:.2f} minute'.format((time.time()-nw)/60))
##############################################################################
#Functional model selection depending on the choosen kernel type
function = [None]*data_number
coeff_mat = [None]*data_number
for i in range(data_number):
    if kernel_type == 1:
        norm_fac = 1e-3
        if data_type[i] == 1:
            function[i] = kernel_functions.pointmass_anomaly_kernel
        else:
            function[i] = kernel_functions.pointmass_disturbance_kernel
    elif kernel_type == 2:
        norm_fac = 1e3
        if data_type[i] == 1:
            function[i] = kernel_functions.poisson_anomaly_kernel
        else:
            function[i] = kernel_functions.poisson_disturbance_kernel
    else:
        norm_fac = 1e2
        if data_type[i] == 1:
            function[i] = kernel_functions.poisson_wavelet_anomaly_kernel
        else:
            function[i] = kernel_functions.poisson_wavelet_disturbance_kernel
            
if control_data == 1:
    if kernel_type == 1 and control_data_type == 1:
        function_control = kernel_functions.pointmass_anomaly_kernel
    elif kernel_type == 1 and control_data_type == 2:
        function_control = kernel_functions.pointmass_disturbance_kernel
    elif kernel_type == 2 and control_data_type == 1:
        function_control = kernel_functions.poisson_anomaly_kernel
    elif kernel_type == 2 and control_data_type == 2:
        function_control = kernel_functions.poisson_disturbance_kernel
    elif kernel_type == 3 and control_data_type == 1:
        function_control = kernel_functions.poisson_wavelet_anomaly_kernel
    else:
        function_control = kernel_functions.poisson_wavelet_disturbance_kernel 
##############################################################################
#Spherical Approximation will be used
lat_sph = [None]*data_number
lon_sph = [None]*data_number
rad_dist = [None]*data_number 
cart_ell = [None]*data_number
for i in range(data_number):       
    # Transformation Ellipsoidal Coordinates to Spherical Coordinates 
    lat_sph[i],lon_sph[i],R = other_functions.ell2sph(lat[i],lon[i],H[i])
    
    rad_dist[i] = (R_bjerhammer*1e3+ H[i])*norm_fac #Spherical Approximation in meter
##############################################################################
#Long wavelength part of gravity calculation using GGM for input data
if res_sel == 0:
    print('\nLong wavelength calculation from selected GGM started...')
    long_wav = [None]*data_number
    nw = time.time()
    for i in range(data_number):
        if data_type[i] == 1:
            functions = gravity_functions.anomaly_from_ggm
        else:
            functions = gravity_functions.disturbance_from_ggm
        # Transformation to cartesian coordinates
        cart_ell[i] = other_functions.geodetic2cartesian(lat[i],lon[i],H[i])
        
        long_wav[i] = other_functions.multiprocessing_model_ggm(C,S,Lmin,cart_ell[i],GM,radius,1,functions,workers)
    
        long_wav[i] = long_wav[i]* (1e5) # Turn to mGal
        residual_data[i] = grav[i] - (long_wav[i]+rtm[i])
        cart_ell[i] = None
        print('\nGGM long wavelength calculation complated. Process time:{:.2f} minute'.format((time.time()-nw)/60))
##############################################################################
if control_data == 1:

    print('\nLong wavelength calculation of external control points started')
    
    if control_data_type == 1:
        functions = gravity_functions.anomaly_from_ggm
    else:
        functions = gravity_functions.disturbance_from_ggm
    
    if res_sel == 0:
        cart_ell_control = other_functions.geodetic2cartesian(lat_control,lon_control,H_control)
        
        long_wav_control = other_functions.multiprocessing_model_ggm(C,S,Lmin,cart_ell_control,GM,radius,1,functions,workers)
        
        long_wav_control = long_wav_control*(1e5) # Turn to mGal
        cart_ell_control = None
        R = None
        
        residual_data_control = grav_control - (long_wav_control+rtm_control)
        
    lat_sph_control,lon_sph_control,R = other_functions.ell2sph(lat_control,lon_control,H_control)
    rad_dist_control = (R_bjerhammer*1e3 + H_control)*norm_fac
gc.collect()
##############################################################################
print('\nk-SRBF algorithm is initialized. It can take some time. Please wait...')
nw = time.time()
lat_srbf,lon_srbf = k_SRBF.k_SRBF(lat_sph,lon_sph,k_init,k_min,min_srbf,max_srbf,R_bjerhammer,workers)
gc.collect()  
print('\nThe data-adaptive network design completed. Process time:{:.2f} minute'.format((time.time()-nw)/60))          
##############################################################################
print('\nThe software trying to determine normal equation matris is singular or not')
print('\nIf normal equation matrix is singular, variance-component estimation will be used')
if depth_choose == 2:
    rad_dist_rbf = np.ones(len(lat_srbf))*(R_bjerhammer-depth_i)*1e3*norm_fac
else:
    rad_dist_rbf = np.ones(len(lat_srbf))*(R_bjerhammer-depth)*1e3*norm_fac
    
for i in range(data_number):
    coeff_mat[i] = coeff_matrix.other_kernel_coeff_mat(rad_dist[i],lat_sph[i],lon_sph[i],0,rad_dist_rbf,function[i],lat_srbf,lon_srbf,workers)
   
regu = other_functions.test_normal_eqution(coeff_mat)
if regu == 0:
    print('\nNormal equation matrix is nonsingular, least-square solution will be performed without regularization')
else:
    print('\nNormal equation matrix is singular, least-square solution will be performed with MCVCE')
coeff_mat = [None]*data_number
gc.collect()
##############################################################################
if depth_choose == 2:
    depth_range = np.arange(depth_i,depth_l+depth_s,depth_s)
    depth_last,gcv = k_SRBF.depth_sel_with_GCV(lat_sph,lon_sph,rad_dist,lat_srbf,lon_srbf,residual_data,R_bjerhammer,function,workers,depth_range,regu,norm_fac,total_data)
else:
    depth_last = depth
print('\nDepth selected as {:.1f}'.format(depth_last))    
gc.collect() 
##############################################################################      
if Lmin_choose == 2:
    rms_control3 = []; Lmin2 = []
    Lmin_range = np.arange(Lmin_i,Lmin_s+Lmin_l,Lmin_s)
    rad_dist_rbf = np.ones(len(lat_srbf))*(R_bjerhammer-depth_last)*1e3*norm_fac
    parameter_poll = np.zeros((len(rad_dist_rbf),len(Lmin_range)))
    sigma_poll = [None]*len(Lmin_range)
    k = 0
    for j in Lmin_range:
        print('\nTesting for {} reduced SRBF. Please wait...'.format(j))
        for i in range(data_number):
            coeff_mat[i] = coeff_matrix.other_kernel_coeff_mat(rad_dist[i],lat_sph[i],lon_sph[i],j,rad_dist_rbf,function[i],lat_srbf,lon_srbf,workers)
        print('\nParameter Estimation started. Please wait...')
        if regu == 0:
            parameter = other_functions.LS(coeff_mat,residual_data)
        else: 
            parameter,sigma = other_functions.MCVCE(coeff_mat,residual_data)   
            sigma_poll[k] = sigma

        parameter_poll[:,k] = parameter
        coeff_mat = [None]*data_number
        gc.collect()
        coeff_mat_test = coeff_matrix.other_kernel_coeff_mat(rad_dist_control,lat_sph_control,lon_sph_control,j,rad_dist_rbf,function_control,lat_srbf,lon_srbf,workers)
       
        estimated_test = np.dot(coeff_mat_test,parameter)
       
        residual_test = residual_data_control - estimated_test
       
        rms_anomaly_test = np.round(np.sqrt(np.sum(residual_test**2)/len(residual_test)),3)
        
        rms_control3 = np.append(rms_control3,float(rms_anomaly_test))
        Lmin2 = np.append(Lmin2,j)
        print('RMS for {} reduced SRBF is {:.3f}'.format(j,rms_anomaly_test))
        coeff_mat_test = None; parameter=None
        k += 1
        if j != Lmin_i:
            if rms_anomaly_test>1.25*rms_control3[0]:
                break
        
    Lmin_last,Lmin_index = other_functions.sel_turn_frm_graph(Lmin_range[0:len(rms_control3)],rms_control3)
    print('Reduced SRBF value selected as {}'.format(Lmin_last))
else:
    Lmin_last = Lmin_2
    print('Reduced SRBF value selected as {}'.format(Lmin_last))
##############################################################################    
if Lmin_choose == 2:  
    parameter = parameter_poll[:,Lmin_index]
    parameter_poll = None
    if regu == 1:
        sigma = sigma_poll[Lmin_index]
        sigma = None
else:
    print('\nFinal Coefficient Matrix Calculation. Please wait...')
    rad_dist_rbf = np.ones(len(lat_srbf))*(R_bjerhammer-depth_last)*norm_fac*1e3
    for i in range(data_number):
        coeff_mat[i] = coeff_matrix.other_kernel_coeff_mat(rad_dist[i],lat_sph[i],lon_sph[i],Lmin_last,rad_dist_rbf,function[i],lat_srbf,lon_srbf,workers)
    
    print('Coefficient matrix calculation done. Process time: {:.2f} minute'.format((time.time()-nw)/60))
    print('\nParameter Estimation started. Please wait...')   
    nw = time.time()
    if regu == 0:
        parameter = other_functions.LS(coeff_mat,residual_data)
    else: 
        parameter,sigma = other_functions.MCVCE(coeff_mat,residual_data) 
    coeff_mat = [None]*data_number
    gc.collect()
    print('\nParameter Estimation completed.Process time: {:.2f} minute'.format((time.time()-nw)/60))
##############################################################################
# External Validation (Control Points)
if control_data == 1:
    print('\nExternal validation for gravity control points started...')
    nw = time.time()
    
    sph_dist_test = other_functions.sph_dist(lat_sph_control,lon_sph_control,lat_srbf,lon_srbf,workers)
    coeff_mat_test = other_functions.multiprocessing_model_coeff_pointmass(rad_dist_control,rad_dist_rbf,Lmin_last,sph_dist_test,function_control,other_functions.coeff_mat_pointmass,workers)
    print('Test coefficient matrix calculation done.Process time:{:.2f} minute'.format((time.time()-nw)/60))
    if res_sel == 0:
        estimated_test = np.dot(coeff_mat_test,parameter)
        estimated_control = estimated_test+long_wav_control+rtm_control
        residual_test = grav_control - estimated_control
    else:
        estimated_control= np.dot(coeff_mat_test,parameter)
        residual_test = residual_data_control - estimated_control
    rms_test = np.sqrt(np.sum(residual_test**2)/len(residual_test))
    coeff_mat_test = None
    print('\nRMS of control poinst: {:.3f} mgal'.format(rms_test))
##############################################################################
if qgeoid == 1:
    print('\nQuasi-geoid estimation started...')
    #GPS/Levelling Point Calculation
    nw = time.time()
    if res_sel == 0:
        data_qgeoid = np.transpose(np.array([qgeoid_lat,qgeoid_lon,qgeoid_H]))
        
        terr_cor_qgeoid = RTM_corrections.terrain_correction_multi(data_qgeoid,dem_file,coarse_file,ref_file,lat1_dem,lat2_dem,lon1_dem,lon2_dem,3,4,1,1,2.67,r1,r2,workers)   
            
        ide_qgeoid,rtm_qgeoid = RTM_corrections.read_rtm(terr_cor_qgeoid)
        ide_qgeoid = ide_qgeoid.astype(int)
        
        os.remove('DEM/'+terr_cor_qgeoid)
    
        qgeoid_lat = qgeoid_lat[ide_qgeoid]; qgeoid_lon = qgeoid_lon[ide_qgeoid]; qgeoid_H = qgeoid_H[ide_qgeoid]
        
        cart_ell_qgeoid = other_functions.geodetic2cartesian(qgeoid_lat,qgeoid_lon,qgeoid_H)
        
        long_wav_quasi_qgeoid = other_functions.multiprocessing_model_ggm(C,S,Lmin,cart_ell_qgeoid,GM,radius,1,gravity_functions.disturbing_potential,workers)

    #Cartesian and Spherical Coordinates of qgeoid

    qgeoid_lat_sph,qgeoid_lon_sph,R_qgeoid = other_functions.ell2sph(qgeoid_lat,qgeoid_lon,qgeoid_H)
    rad_dist_qgeoid = (R_bjerhammer*1e3+ qgeoid_H)*norm_fac

    normal_gravity_qgeoid = gravity_functions.normal_gravity80(qgeoid_lat) - 0.3086e-5*qgeoid_H# Normal gravity on telluroid
    
    sph_dist_qgeoid =other_functions.sph_dist(qgeoid_lat_sph,qgeoid_lon_sph,lat_srbf,lon_srbf,workers) 

    if kernel_type == 1:
        functions_2 = kernel_functions.pointmass_kernel
    elif kernel_type == 2: 
        functions_2 = kernel_functions.poisson_kernel
    else:
        functions_2 = kernel_functions.poisson_wavelet_kernel
        
    coeff_mat_qgeoid = other_functions.multiprocessing_model_coeff_pointmass(rad_dist_qgeoid,rad_dist_rbf,Lmin_last,sph_dist_qgeoid,functions_2,other_functions.coeff_mat_pointmass,workers)
    
    if res_sel == 0:
        estimated_disturbing_qgeoid = (np.dot(coeff_mat_qgeoid,parameter)/(1e5))/norm_fac
            
        estimated_qgeoid = ((long_wav_quasi_qgeoid + estimated_disturbing_qgeoid) / normal_gravity_qgeoid) + rtm_qgeoid
    else:
        estimated_qgeoid = ((np.dot(coeff_mat_qgeoid,parameter)/(1e5))/norm_fac) / normal_gravity_qgeoid
        
    coeff_mat_qgeoid = None; sph_dist_qgeoid = None
    print('Quasi-geoid Points Calculation Done. Process time: {:.2f} minute'.format((time.time()-nw)/60))
else:
    pass
##############################################################################
with open('output/Results.txt','w') as sonuclar:
    if kernel_type == 1:
        sonuclar.write('Used Kernel type: Point mass kernel\n')
    elif kernel_type == 2:
        sonuclar.write('Used Kernel type: Poisson kernel\n')  
    else:
        sonuclar.write('Used Kernel type: Poisson Wavlet (order 3) kernel\n')
        
    sonuclar.write('Number of Points:  {}\n'.format(total_data))
    sonuclar.write('Number of SRBF:   {}\n'.format(len(lat_srbf)))
        
    if control_data == 1:
        sonuclar.write('Number of Control Points: {}\n'.format(len(lat_control)))
    sonuclar.write('Degree of GGM (long-wavelength): {}\n'.format(Lmin))
    
    if regu == 1:
        sonuclar.write('\nEstimated sigma values\n')
        for i in range(data_number):
            sonuclar.write('{:.0f}.st data sigma value: {:5.5f}\n'.format((i+1),sigma[i][-1]))
        sonuclar.write('Sigma valued of estimated coefficients: {:5.5f}\n'.format(sigma[-1][-1]))
    
    if control_data == 1:
        sonuclar.write('\nControl data statistic\n')
        sonuclar.write('RMS of control points residual   : {:5.3f} mgal\n'.format(rms_test))
        sonuclar.write('Mean of control points residual  : {:5.3f} mgal\n'.format(np.mean(residual_test)))
        sonuclar.write('Min of control points residual   : {:5.3f} mgal\n'.format(np.min(residual_test)))
        sonuclar.write('Max of control points residual   : {:5.3f} mgal\n'.format(np.max(residual_test)))
        sonuclar.write('STD of control points residual   : {:5.3f} mgal\n\n'.format(np.std(residual_test)))  
    else: 
        pass
    
    if depth_choose == 2:   
        sonuclar.write('{:>17s}\n'.format('Tested Depth'))
        sonuclar.write('{:>7s} {:>12s}\n'.format('Depth', 'GCV'))
        for i in range(len(depth_range)):
            sonuclar.write('{:>5.0f}    {:>15.5e}\n'.format(depth_range[i],gcv[i]))
            
    if Lmin_choose == 2:
        sonuclar.write('\n\n{:>17s}\n'.format('Tested Lmin'))
        sonuclar.write('{:>7s} {:>25s}\n'.format('Lmin', 'RMS at control points'))
        for i in range(len(Lmin2)):
            sonuclar.write('{:>7.0f} {:>15.3f}\n'.format(Lmin2[i],rms_control3[i]))
    
    sonuclar.write('\n\nFinal Parameters')
    sonuclar.write('\nGrid depth: {:.1f}'.format(depth_last))
    sonuclar.write('\nLmin (reduced SRBF) value: {:.0f}'.format(Lmin_last))
            
if control_data==1:
    with open('output/Control Points.txt','w') as sonuclar:   
        sonuclar.write('{:<13s} {:<13s} {:<13s} {:<10s} {:<13s}\n'.format('ID','Latitude','Longitude','Height','Estimated Value (mGal)'))
        for i in range(len(lat_control)):
            sonuclar.write('{:<12.0f} {:>3.6f} {:>14.6f} {:>10.3f} {:>18.3f}\n'.format(i,lat_control[i],lon_control[i],H_control[i],estimated_control[i]))
else:
    pass
                
if qgeoid == 1:       
    with open('output/Height_anomaly.txt','w') as sonuclar:  
        sonuclar.write('{:<13s} {:<13s} {:<13s} {:<10s} {:<13s}\n'.format('ID','Latitude','Longitude','Height','Quasi-Geoid Ondulation (m)'))
        for i in range(len(qgeoid_lat)):
            sonuclar.write('{:<12.0f} {:>3.6f} {:>14.6f} {:>10.3f} {:>18.3f}\n'.format(i,qgeoid_lat[i],qgeoid_lon[i],qgeoid_H[i],estimated_qgeoid[i]))
else:
    pass 

if kernel_type in (1,3):
    parameter = parameter / ((1e-5) /norm_fac**2)
    unit = '(m^3/s^2)'
else:
    parameter = parameter / ((1e-5) /norm_fac)
    unit = '(m^2/s^2)'
    
with open('output/SRBFs','w') as sonuclar:
    sonuclar.write('{:<13s} {:<13s} {:<13s} {:<16s} {:<13s}\n'.format('ID','Latitude','Longitude','Depth(km)','Magnitude '+unit))
    for i in range(len(lat_srbf)):
        sonuclar.write('{:<12.0f} {:>3.6f} {:>14.6f} {:>9.0f} {:>30.3f}\n'.format(i,lat_srbf[i],lon_srbf[i],depth_last,parameter[i]))    
gc.collect()
print('All process done. Time:{:.2f} minute'.format((time.time()-nw_first)/60))
