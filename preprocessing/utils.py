import glob
import pandas as pd

import torch
import os
import matplotlib.pyplot as plt
from nmrsim import Multiplet
from random import randint, uniform

import time

def folder2tensors(data_dir, high_res):
    # Data loading starting with list of csv strings
    files = glob.glob(os.path.join(data_dir, f'{high_res}MHz', 'spectral_data_*.csv'))

    y_low = [] # Establishes a list for 60 MHz data
    y_high = [] # Establishes a list for 400 MHz data

    print("Reading generated CSVs and putting it into tensors")
    for i, file in enumerate(files): # For loop for each file in files
        print(f"{i+1}/{len(files)}", end='\r')
        df = pd.read_csv(file) # Reads each into a pandas dataframe
        array_low = df['60MHz_intensity'].to_numpy() # Takes 60MHz intensity to np
        array_high = df[f'{high_res}MHz_intensity'].to_numpy() # Takes 400MHz intensity to np
        y_low.append(array_low) # Appends all arrays to 60MHz list
        y_high.append(array_high) # Appends all arrays to 400MHz list
        
    # Creates a 60 MHz tensor from list, converts to float, unsqueezes to have shape (n, 1, 5500)
    tensor_low = torch.Tensor(y_low).float().unsqueeze(1)
    # Creates a high resolution MHz tensor from list, converts to float, unsqueezes to have shape (n, 1, 5500)
    tensor_high = torch.Tensor(y_high).float().unsqueeze(1)

    torch.save(
        {'low': tensor_low, 'high': tensor_high}, 
        os.path.join(data_dir, f'{high_res}MHz.pt'),
    )
    return tensor_low, tensor_high

def NMR_Signal_Generator(spectrometer_frequencies):
    '''
    Takes input of list of 2 spectrometer frequencies in list format and
    generates a list of NMR signals using nmrsim library. An NMR signal given
    by this function is described by 1-4 (random), a 2Hz linewidth, a chemical
    shift from 0.5 to 9, a J coupling of 3 to 15Hz, a multiplicity of 0 to 5
    
    Parameters
    ----------
        spectrometer_frequencies: List
            List of 2 spectrometer frequencies
            
    Output
    -------
        signals_list: List
            List of NMR signals of the two resolutions
    
    
    '''
    # Assigning number of protons defined by peak integral
    integral = randint(1,4)
    #Assigning the linewidth of an NMR peak in the spectrum
    linewidth_hz = 2
    # Randomly return a floating point chemical shift assignment
    chemical_shift = uniform(0.5,9)
    # Randomly return a floating point coupling (J coupling) frequency
    coupling = uniform(3,15)
    # Randomly select a multiplicity (peak splitting)
    multiplicity = randint(0,5)
    # Generate a list of NMR signals using the Multiplet function of nmrsim
    signals_list = [(Multiplet(chemical_shift * frequency, integral, 
                               [(coupling,multiplicity)], linewidth_hz)) for 
                                frequency in spectrometer_frequencies]
    return signals_list


def create_random_spectra(num_peaks, res_1, res_2):
    '''
    Creates random NMR spectra given number of peaks desired, resolution 1
    and resolution 2
    
    Parameters
    -----------
        num_peaks: int
            Number of peaks in random spectrum
        res_1: int
            First desired resolution in MHz
        res_2: int
            Second desired resolution in MHz
    
    Output
    -------
        x_res_1: float
            Chemical shift in ppm of spectrum 1
        y_res_1: float
            Intensity of spectrum 1
        x_res_2: float
            Chemical shift in ppm of spectrum 2
        y_res_2: float
            Intensity of spectrum
    

    '''
    # Assigns desired NMR frequencies to a list
    spectrometer_frequencies = [res_1,res_2]
    # Creates blank dataframe with desired NMR frequencies column
    spectral_data = pd.DataFrame(columns=[f'{str(res_1)}MHz', f'{str(res_2)}MHz'])
    
    # Generates NMR Signals from NMRSIM using established list, generates
    # spectra and puts them in spectral_data frame
    for i in range(0,num_peaks):
        signals_list = NMR_Signal_Generator(spectrometer_frequencies)
        spectral_data.loc[len(spectral_data)] = signals_list

    # Defines spectrum object, from Multiplet class
    # Multiplets taken from spectral_data frequency df
    spectrum_res_1 = Multiplet(0,0.5,[],2) 
    for multiplet in spectral_data[f'{str(res_1)}MHz']:
        spectrum_res_1 += multiplet
        
    # Process repeated for other frequency
    spectrum_res_2 = Multiplet(0,0.5,[],2)
    for multiplet in spectral_data[f'{str(res_2)}MHz']:
        spectrum_res_2 += multiplet
    
    # Normalize the spectrometer frequencies and have n number of points on plots
    spectrum_res_1.vmin = -0.5 * spectrometer_frequencies[0]
    spectrum_res_1.vmax = 10.5 * spectrometer_frequencies[0]
    x_res_1, y_res_1 = spectrum_res_1.lineshape(points=5500)

    spectrum_res_2.vmin = -0.5 * spectrometer_frequencies[1]
    spectrum_res_2.vmax = 10.5 * spectrometer_frequencies[1]
    x_res_2, y_res_2 = spectrum_res_2.lineshape(points=5500)

    return x_res_1, y_res_1, x_res_2, y_res_2


def write_nmr_spectra(index, num_peaks, x_res_1, y_res_1, x_res_2, y_res_2, res_1, res_2, data_dir):
    '''
    Writes generated NMR spectra to csv files
    
    Parameters
    -----------
        index: int
            Label as integer of NMR spectrum generated in a series
        num_peaks: int
            Specifies how many peaks are desired in spectrum
        x_res_1: float
            Output of write spectrum, chemical shift of spectrum 1
        x_res_2: float
            Output of write spectrum, chemical shift of spectrum 2
        y_res_1: float
            Output of write spectrum, intensity of spectrum 1
        y_res_2: float
            Output of write spectrum, intensity of spectrum 2
        res_1: int
            Resolution in MHz of spectrum 1
        res_2: int
            Resolution in MHz of spectrum 2
    
    Output
    -------
        filename: csv
            A csv containing the x, y data of spectra 1 and 2
    
    
    '''
    # Saving data to file
    sf = [res_1,res_2]
    x_ppm_res_1 = x_res_1/sf[0]
    x_ppm_res_2 = x_res_2/sf[1]
    spectral_data = pd.DataFrame(columns=[f'{str(res_1)}MHz_ppm',
                                        f'{str(res_1)}MHz_intensity', 
                                        f'{str(res_2)}MHz_ppm',
                                        f'{str(res_2)}MHz_intensity'])
    spectral_data[f'{str(res_1)}MHz_ppm'] = x_ppm_res_1
    spectral_data[f'{str(res_1)}MHz_intensity'] = y_res_1
    spectral_data[f'{str(res_2)}MHz_ppm'] = x_ppm_res_2
    spectral_data[f'{str(res_2)}MHz_intensity'] = y_res_2
    filename="spectral_data"+ "_"+str(num_peaks).zfill(2)+ "_" +str(index).zfill(5)+".csv"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(os.path.join(data_dir, f"{res_1}MHz")):
        os.makedirs(os.path.join(data_dir, f"{res_1}MHz"))


    spectral_data.to_csv(os.path.join(data_dir, f"{res_1}MHz", filename))