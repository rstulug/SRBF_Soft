#!/usr/bin/env pypy3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:33:46 2019

@author: Rasit ULUG
Middle East Technical University
"""
import os
import subprocess
import numpy as np
from multiprocessing import Pool

"""
Residual Terrain Correction using Gravsoft TC module.
We use paralel processing to decrease computation time for huge data.

For detailed information please look:
    
Forsberg, R., & Tscherning, C. C. (2008). GRAVSOFT. Geodetic gravity field 
modelling programs (overview manual).

"""
def terrain_correction(data,dem_file,average_dem_file,ref_dem_file,lat1,lat2,lon1,lon2,typ,rtm,stat,sub,dens,r1,r2):
#    typ=1 disturbance 
#    typ=2 deflections
#    typ=3 height anomaly
#    typ=4 dg,ksi,eta
#    typ=5 gravity anomaly
#    typ=6 anomaly,ksi,eta,ha
#    typ=7 tzz
#    typ=8 txx,tyy,tzz
#    typ=9 all gradients
    try:
        file_name = dem_file.split('.')
        terrain_corr = file_name[0] +'_rtm.'+file_name[1]
    except:
        terrain_corr = file_name[0] +'_rtm.'
    input_name = 'DEM/'+data+'\n'+'DEM/'+dem_file+'\n'+'DEM/'+average_dem_file+'\n'+'DEM/'+ref_dem_file+'\n'+'DEM/'+terrain_corr+'\n'+\
    str(typ)+' '+str(rtm)+' '+str(stat)+' '+str(sub)+' '+str(dens)+'\n'+\
    str(lat1)+ ' '+str(lat2)+ ' '+ str(lon1) + ' ' + str(lon2) + '\n' + str(r1)+ ' ' + str(r2) +'\n'
    process = subprocess.Popen(['bin/tc'],stdin=subprocess.PIPE)
    process.communicate(str.encode(input_name))
    process.stdin.close()
    return(terrain_corr)

#   Below funcions necessary for parallel processing of terrain correction subprocess. 
#   For huge data, it will decrease the computation time depending on the number of cores
def ter_cor(args):
    data = args[0]; dem_file=args[1]; average_dem_file=args[2]; ref_dem_file=args[3]
    lat1 = args[4]; lat2 = args[5]; lon1 = args[6]; lon2 = args[7]
    typ=args[8]; rtm=args[9]; stat=args[10]; sub=args[11];dens=args[12]
    r1 = args[13];r2 = args[14];order=args[15]
    
    file_name = dem_file.split('.')
    try:
        terrain_corr = file_name[0]+'_'+str(order) +'_rtm.'+file_name[1]
    except:
        terrain_corr = file_name[0]+'_'+str(order) +'_rtm.'
    input_name = 'DEM/'+data+'\n'+'DEM/'+dem_file+'\n'+'DEM/'+average_dem_file+'\n'+'DEM/'+ref_dem_file+'\n'+'DEM/'+terrain_corr+'\n'+\
    str(typ)+' '+str(rtm)+' '+str(stat)+' '+str(sub)+' '+str(dens)+'\n'+\
    str(lat1)+ ' '+str(lat2)+ ' '+ str(lon1) + ' ' + str(lon2) + '\n' + str(r1)+ ' ' + str(r2) +'\n'
    process = subprocess.Popen(['bin/tc'],stdin=subprocess.PIPE)
    process.communicate(str.encode(input_name))
    process.stdin.close()
 
    return(terrain_corr)

def terrain_correction_multi(data,dem_file,average_dem_file,ref_dem_file,lat1,lat2,lon1,lon2,typ,rtm,stat,sub,dens,r1,r2,workers):
    indis = np.linspace(0,len(data),workers,endpoint=True,dtype=int)
    for i in range(workers-1):
        with open('DEM/data'+str(i)+'.txt','w') as sonuclar:
            for k in range(indis[i],indis[i+1]):
                sonuclar.write('{:4d}  {}\n'.format(k,str(data[k,:]).strip('[]')))
    
    args = [None]*(workers-1)
    
    for i in range(workers-1):
        args[i] = ['data'+str(i)+'.txt',dem_file,average_dem_file,ref_dem_file,lat1,lat2,lon1,lon2,typ,rtm,stat,sub,dens,r1,r2,i]
    
    pool = Pool()
    results = pool.map_async(ter_cor,args).get()
    pool.close()
    pool.terminate()
    pool.join()
    
    filenames = results
    last_rtm = 'lastrtm'
    with open('DEM/'+last_rtm, 'w') as outfile:
        for fname in filenames:
            with open('DEM/'+fname) as infile:
                for line in infile:
                    outfile.write(line)         
    
    for i in range(workers-1):
        os.remove('DEM/'+'data'+str(i)+'.txt')
        os.remove('DEM/'+filenames[i])   
    return(last_rtm)

def read_rtm(filename):
    basePath = 'DEM/'+filename
    rnxfile = open(basePath)
    lines = rnxfile.read().splitlines()
    rtm = np.zeros(len(lines))
    ide = np.zeros(len(lines))
    for i in range(len(lines)):
        line = lines[i].split()
        rtm[i] = float(line[4])
        ide[i] = float(line[0])
    return(ide,rtm)