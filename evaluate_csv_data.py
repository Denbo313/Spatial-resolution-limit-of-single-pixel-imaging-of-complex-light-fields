# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:21:26 2024

@author: Dennis Scheidt; dennis.scheidt@correo.nucleares.unam.mx
"""

#%% 0) define helper functions and import libraries

import os 
import numpy as np
import pandas as pd
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
import pandas as pd

def NCC_real(A,B):
    """
    calculates the normalized cross correlation of the real valued images A and B according to:
    A. Kaso, “Computation of the normalized cross-correlation by fast fourier transform,” PLOS ONE 13, e0203434 (2018).365

    """
    num = (np.dot(A.flatten(),B.flatten()))
    num = (A * B).sum()
    denom = abs(np.sqrt((A**2).sum()) * np.sqrt((B**2).sum()))
    denom = np.sqrt((abs(A)**2).sum())*np.sqrt((abs(B)**2).sum())
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    return (num/denom)


def sublist(ls,key):
    """
    helper function to extract a list from a list that matches a certain criterion
    """
    return [match for match in ls if key in match]


def reconstruct_field(vecs, dat):
    """
    calculates the copmlex vector from a SPI measurement with 3 phase step interferometry
    """
    x = np.zeros(dat.shape)
    
    # for every phase step reconstruct the corresponding field
    for i in range(3):
        x[:,i] = vecs @ dat[:,i]
        
    phasor = -1/3 * (x[:,1] + x[:,2] - 2*x[:,0]) + 1j/np.sqrt(3) * (x[:,2] - x[:,1])
    return phasor
        
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

def reduce_vec(a):
    """
    helper function to reduce the length of arrays that contain a lot of zeros
    """
    return a[a != 0]

#%% 1) read the experimental and simulated data

input_direc = r'data_fields\\'

filelist = sorted(os.listdir(input_direc), key=len)

### IMPORTANT for saving the reconstructed data to your local directory, put the flag to TRUE
export_data = True
output_direc =  r'field_reconstruction\\'

### visualization flag
plot_data = False # set to True in order to plot the data

nn = 32 # size of the field grid

rec_amps = []
rec_phases = []
rec_fields = []

#%% 2) reconstruct phase and amplitude of the experimental and simulated data data

for file in filelist:
    # data = np.loadtxt(input_direc + file)
    df = pd.read_csv(input_direc +file)
    dat = df.values[:,1:4]
    
    if '64' in file:
        N = 64
    elif '256' in file:
        N = 256
    elif '1024' in file:
        N = 1024
    
    n = int(np.sqrt(N))
    sx = nn//n
    
    if 'canon' in file:
        vecs = np.eye(N)
    else:
        vecs = hadamard(N)
    
    phasor = reconstruct_field(vecs,dat)
    
    rec_amp = np.abs(phasor)#.reshape((n,n))
    rec_amp = vec_2_mask(rec_amp, (nn,nn), (sx,sx)) # reshape the reconstructed array to the original (32x32) grid
    rec_amp = rec_amp/rec_amp.max()
    
    rec_phase = np.angle(phasor)
    rec_phase =  vec_2_mask(rec_phase, (nn,nn), (sx,sx)) # reshape the reconstructed array to the original (32x32) grid
    # normalize to 2pi to yield input field
    rec_phase = (rec_phase)%(2*np.pi)

        
    ##visualization of reconstructed data
    if plot_data:
        plt.subplot(1,2,1)
        plt.imshow(rec_amp,cmap = 'gray')
        plt.title('reconstructed amplitude')
        plt.subplot(1,2,2)
        plt.imshow(rec_phase,cmap = 'bone')
        plt.title('reconstructed phase')
        plt.show()
    
    # save/export data to local file, if flag is set:
    if export_data:
        plt.imsave(output_direc + 'Amp_rec_'+ file[:-4] +'.png',rec_amp,cmap = 'gray')
        plt.imsave(output_direc + 'Phase_rec_'+ file[:-4] +'.png',rec_phase,cmap = 'bone')
        
        plt.imsave(output_direc + 'Amp_rec_'+ file[:-4] +'.eps',rec_amp,cmap = 'gray')
        plt.imsave(output_direc + 'Phase_rec_'+ file[:-4] +'.eps',rec_phase,cmap = 'bone')
        
        
    # store data in a list for later evaluation of the NCC
    rec_amps.append(rec_amp)
    rec_phases.append(rec_phase)
    rec_fields.append(rec_amp * np.exp(1j * rec_phase))
    

#%% 3) evaluate the cross-talk by using the NCC (field, amplitude and phase)
input_direc_fields = r'C:\Users\D S\PhD\paper_parceval\fields - Copy\exp\input_fields'
amp_gauss = np.loadtxt(input_direc_fields + r'\gauss_amplitude.csv').reshape((n,n))
amp_dog = np.loadtxt(input_direc_fields + r'\dog_amplitude.csv').reshape((n,n))
phase_boat = np.loadtxt(input_direc_fields + r'\boat_phase.csv').reshape((n,n))

l = len(filelist)

# quick and dirty solution to make the evaluation of the different fields and their components (amplitude and phase) more clear
# this code could be done in a single array, but this will make the assignement of single values more complex for the reader
ncc_dog_exp_crosstalk = np.zeros(l)
ncc_dog_sim_crosstalk = np.zeros(l)
ncc_gauss_exp_crosstalk = np.zeros(l)
ncc_gauss_sim_crosstalk = np.zeros(l)

for field, amp, phase, name, i in zip(rec_fields, rec_amps, rec_phases,filelist, range(l)):
    if 'Dog' in name:
        # decision making to extract the correct values
        if 'exp' in name:
            # normalize values to make them comparable
            A = rec_amps[2]
            B = amp
            ncc_dog_exp_crosstalk[i] = NCC_real(A,B)
                      
        elif 'sim' in name and 'hadamard' in name:
            A = rec_amps[14]
            B = amp
            ncc_dog_sim_crosstalk[i] = NCC_real(A,B)
       
    if 'Gauss' in name:
        if 'exp' in name:
            A = rec_amps[5]
            B = amp
            ncc_gauss_exp_crosstalk[i] = NCC_real(A,B)
                       
        elif 'sim' in name and 'hadamard' in name:
            A = rec_amps[17]
            B = amp
            ncc_gauss_sim_crosstalk[i] = NCC_real(A,B)
            
            

# delete the zero entries from the arrays
ncc_dog_exp_crosstalk = reduce_vec(ncc_dog_exp_crosstalk)
ncc_dog_sim_crosstalk = reduce_vec(ncc_dog_sim_crosstalk)
ncc_gauss_exp_crosstalk = reduce_vec(ncc_gauss_exp_crosstalk)
ncc_gauss_sim_crosstalk = reduce_vec(ncc_gauss_sim_crosstalk)


df=pd.DataFrame({"Gauss simulation": ncc_gauss_sim_crosstalk,
                 "Gauss experiment": ncc_gauss_exp_crosstalk,
                 "Dog simulation": ncc_dog_sim_crosstalk,
                 "Dog experiment": ncc_dog_exp_crosstalk})

df.index = ['8 x 8', '16 x 16', '32 x 32']

df.to_csv('NCC_results.csv')