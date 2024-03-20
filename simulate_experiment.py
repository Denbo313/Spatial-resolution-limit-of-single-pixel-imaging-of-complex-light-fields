# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 23:15:34 2024

@author: Dennis Scheidt; dennis.scheidt@correo.nucleares.unam.mx
"""

#%% 0) Import functions and define parameters
import numpy as np
from numpy.fft import fft2, fftshift 
from scipy.linalg import hadamard
import pandas as pd

pi = np.pi

def prism_phase(X,Y,kx,ky):
    """
    calculate the prism phase for a given grid (X,Y) 
    with a displacement in x and y given by kx and ky, respectively
    """
    return (X*kx + Y *ky) % (2*pi)

def vec_2_mask(vec,mask_shape,superpixel_shape):
    """
    take a 1d vector and project it into a 2d distribution
    """
    mask = np.zeros(mask_shape)
    [n,m] = np.array(mask_shape) // np.array(superpixel_shape)
    
    arr_x = np.linspace(0,mask_shape[0], n+1, dtype = int)
    arr_y = np.linspace(0,mask_shape[1], m+1, dtype = int)
    
    arr_x1 = arr_x[0:-1]; arr_x2 = arr_x[1:]
    arr_y1 = arr_y[0:-1]; arr_y2 = arr_y[1:]
    iterator = 0
    for i1, i2 in zip(arr_x1,arr_x2):
        for j1,j2 in zip(arr_y1,arr_y2):
            mask[i1:i2,j1:j2] = vec[iterator]
            iterator += 1
    return mask

input_direc = r'input_fields'
output_direc = r'data_fields\\'

# number of measurements (8x8, 16x16 and 32x32)
Ns = [64,256,1024]
n = 32 #dimension of the grid

offset = np.arange(3) * 2*pi/3

wid = 1.5
ns = 32
x = np.linspace(-wid,wid,ns)
y = x.copy()
X,Y = np.meshgrid(x,y)

kx = 0; ky = 0
xm = 16; ym = 16
prism = prism_phase(X, Y, kx, ky)

#%% 1) load images from csv data and prepare fields
amp_gauss = np.loadtxt(input_direc + r'\gauss_amplitude.csv').reshape((n,n))
amp_dog = np.loadtxt(input_direc + r'\dog_amplitude.csv').reshape((n,n))
phase_boat = np.loadtxt(input_direc + r'\boat_phase.csv').reshape((n,n))

field_gauss = amp_gauss * np.exp(1j * phase_boat)
field_dog = amp_dog * np.exp(1j * phase_boat)
fields = [field_gauss, field_dog]
amps = [amp_gauss, amp_dog]
reference = np.ones((n,n)) * np.exp(1j * prism)

names = ['sim_Gauss_','sim_Dog_']
names_vec = ['canon_','hadamard_']

#%% 2) simulate the experiment

OFF = 2
OFF = 0

for N in Ns: #iterate over resolutions
    
    nx = int(np.sqrt(N)) #resolution of the grid dependent on the measurement basis 
    sx = n//nx # size of a superpixel
    
    M_had = hadamard(N)
    M_can = np.eye(N)
    Ms = [M_can,M_had]
    
    for vecs, name_vec in zip(Ms,names_vec): # iterate over measurement bases
        
        vecn = abs(vecs - 1) // 2 # for the Hadamard matrix: use only the negative entries in the measurement vector
        
        # flag for handling different bases
        if 'canon' in name_vec:
            m_flag = True
        else:
            m_flag = False
        
        for amp, a_name in zip(amps,names): # iterate over the different amplitudes
            
            data = np.zeros((N,3))
            name = 'data_' + a_name + name_vec + str(N) + '.csv' 
            
            for it in range(N): # iterate over the basis vectors
                
                # choose corresponding vector
                if m_flag:
                    vec = vecs[it]
                else:
                    vec = vecn[it]
                    
                # reshape the 1d vector to a 2d mask for the field
                mask = vec_2_mask(vec, (n,n), (sx,sx))
                    
                for j, off in zip(range(3), offset): # iterate over each phase step
                    
                    if m_flag:
                        # activates only the corresponding superpixel of the canonical basis
                        wave = mask * amp * np.exp(1j * ( (phase_boat + prism + off + OFF) % (2*pi) ))
                    else:
                        # applies the Hadamard basis to the phase
                        wave = amp * np.exp(1j * ( (phase_boat + prism + pi * mask + off + OFF) % (2*pi) ))
                    
                    # calculate the intensity in the focused spot
                    intensity = abs(fftshift(fft2(wave +reference)))**2 
                    data[it,j] = intensity[xm,ym]
            
            #%% 3) save the data as CSV 
            df = pd.DataFrame(data, columns=['0 pi','2/3 pi','4/3 pi'])
            df.to_csv(output_direc + name)
    

