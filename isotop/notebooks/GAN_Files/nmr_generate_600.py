#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[69]:


# Here we import all the necessary libraries to generate NMR spectra
import matplotlib.pyplot as plt
from nmrsim import Multiplet
from random import randint, uniform
import pandas


# In[75]:


# Change these variables to change resolutions of spectra generate
# Notation is typically: res_1=higher and res_2=lower (integer input)
res_1=int(input('Input an Integer for High Resolution Spectrum: '))
res_2=int(input('Input an Integer for Lower Resolution Spectrum: '))
# Change this variable to specify number of data points to generate
num_spec=int(input('Input an Integer for Quantity of Spectra Generated: '))

# In[76]:


# Function to generate NMR signals based on input argument of spectrometer freq
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


# In[77]:


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
    spectral_data = pandas.DataFrame(columns=[f'{str(res_1)}MHz', f'{str(res_2)}MHz'])
    
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


# In[78]:


def write_nmr_spectra(index, num_peaks, x_res_1, y_res_1, x_res_2, y_res_2, res_1, res_2):
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
    spectral_data = pandas.DataFrame(columns=[f'{str(res_1)}MHz_ppm',
                                        f'{str(res_1)}MHz_intensity', 
                                        f'{str(res_2)}MHz_ppm',
                                        f'{str(res_2)}MHz_intensity'])
    spectral_data[f'{str(res_1)}MHz_ppm'] = x_ppm_res_1
    spectral_data[f'{str(res_1)}MHz_intensity'] = y_res_1
    spectral_data[f'{str(res_2)}MHz_ppm'] = x_ppm_res_2
    spectral_data[f'{str(res_2)}MHz_intensity'] = y_res_2
    filename=f"spectral_data/{str(res_1)}MHz/"+"spectral_data"+ "_"+str(num_peaks).zfill(2)+ "_" +str(index).zfill(5)+".csv"
    spectral_data.to_csv(filename)


# In[73]:


import time


# In[79]:


# num_peaks = Number of Peaks that you want in your spectra
# j = how many spectra do you want.
x = time.time()

for j in range (0,num_spec):
    num_peaks = randint(5,16)
    #print(num_peaks)
    x_res_1, y_res_1, x_res_2, y_res_2 = create_random_spectra(num_peaks, res_1, res_2)
    write_nmr_spectra(j, num_peaks, x_res_1, y_res_1, x_res_2, y_res_2, res_1, res_2)

print(f'Time Elapsed: {round(time.time()-x, 5)} seconds')    


# In[ ]:

