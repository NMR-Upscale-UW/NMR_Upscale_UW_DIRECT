
import glob
#Checking how many files are in repository for training, testing, and validation
files = glob.glob('/home/fostooq/NMR_Upscale_UW_DIRECT/sprectra_data/400MHz/spectral_data_*.csv')
print('Total number of files: ', len(files))
