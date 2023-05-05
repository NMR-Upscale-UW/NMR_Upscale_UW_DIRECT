'''
This module generates different NMR spectra data and save them to csv fiels.
'''
# Import essential packages
from utils import create_random_spectra, write_nmr_spectra, folder2tensors
import time
import pandas as pd
import os

from random import randint
import argparse
import numpy as np
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--low_resolution", type=int, default=60, help='The lower resolution to generate')
    parser.add_argument("--high_resolution", type=int, default=400, help='The higher resolution to generate')
    parser.add_argument("--num_spec", type=int, default=100, help='The number of data points to generate')
    parser.add_argument("--data_dir", type=str, default='./data', help='Directory to save the data')

    args = parser.parse_args()

    # Change these variables to change resolutions of spectra generate
    # Notation is typically: res_1=higher and res_2=lower (integer input)
    res_1=args.high_resolution
    res_2=args.low_resolution
    num_spec=args.num_spec
    # j = how many spectra do you want.
    x = time.time()

    print(f"Generating {num_spec} samples")
    # Create spectra data
    for j in range(num_spec):
        print(f"Sample: {j+1}/{num_spec}", end='\r')
        # num_peaks = Number of Peaks that you want in your spectra
        num_peaks = randint(5,16)
        #print(num_peaks)
        x_res_1, y_res_1, x_res_2, y_res_2 = create_random_spectra(num_peaks, res_1, res_2)
        write_nmr_spectra(j, num_peaks, x_res_1, y_res_1, x_res_2, y_res_2, res_1, res_2, data_dir=args.data_dir)

    print(f'Time Elapsed: {round(time.time()-x, 5)} seconds')    

    print(f'Now running featurization code...')    
    # Store generated data
    ofilenames = glob.glob(os.path.join(args.data_dir, f"{res_1}MHz/*.csv"))
    ofilenames.sort()
    data_temp=pd.read_csv(ofilenames[0])
    df_res_2 = np.zeros(shape=(len(ofilenames), len(data_temp.index)))
    df_res_1 = np.zeros(shape=(len(ofilenames), len(data_temp.index)))

    x= data_temp[f"{res_1}MHz_ppm"].to_numpy()
    # Data conversion
    for index, f in enumerate(ofilenames):
        data=pd.read_csv(f)
        df_res_2[index]=data[f"{res_2}MHz_intensity"].to_numpy()
        df_res_1[index]=data[f"{res_1}MHz_intensity"].to_numpy()
    dfres_2 = pd.DataFrame(df_res_2,columns=x)
    dfres_1 = pd.DataFrame(df_res_1,columns=x)
    dfres_2.to_csv(os.path.join(args.data_dir, f"{res_1}MHz/*.csv"), index=False)
    dfres_1.to_csv(os.path.join(args.data_dir, f"{res_1}MHz/*.csv"), index=False)
    

    print("Now preprocess the folders and turn them into tensors")
    folder2tensors(args.data_dir, args.high_resolution)
