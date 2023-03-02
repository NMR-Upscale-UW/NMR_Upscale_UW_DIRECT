#!/usr/bin/env python
# coding: utf-8

# import pandas as pd
# import os
# import numpy as np
# This sheets converts the individual NMR Spectra data into featurization sheets
# 1. Read all the files.
# 2. Take the Y values of res_2 MHz and fill the data into column vectors
# 3. Take the X values of res_2 MHz and create column vectors
# 4. Run it for all the files in the directory
# 5. Take the final dataframe and save it as csv
# 6. Do it for both res_2 MHz and res_1 MHz

# In[18]:


import pandas as pd
import os
import numpy as np
import glob


# In[22]:


# Change these variables to change resolutions of spectral target
# Notation is typically: res_1=higher and res_2=lower (integer input)
res_1 = 1200
res_2 = 60


# In[20]:


ofilenames = glob.glob(f"spectral_data/{res_1}MHz/*.csv")
ofilenames.sort()
ofilenames


# In[21]:


data_temp=pd.read_csv(ofilenames[0])
df_res_2 = np.zeros(shape=(len(ofilenames), len(data_temp.index)))
df_res_1 = np.zeros(shape=(len(ofilenames), len(data_temp.index)))

x= data_temp[f"{res_1}MHz_ppm"].to_numpy()

for index, f in enumerate(ofilenames):
    data=pd.read_csv(f)
    df_res_2[index]=data[f"{res_2}MHz_intensity"].to_numpy()
    df_res_1[index]=data[f"{res_1}MHz_intensity"].to_numpy()
dfres_2 = pd.DataFrame(df_res_2,columns=x)
dfres_1 = pd.DataFrame(df_res_1,columns=x)
dfres_2.to_csv(f"spectral_data/{res_1}MHz/*.csv", index=False)
dfres_1.to_csv(f"spectral_data/{res_1}MHz/*.csv", index=False)
# For loop, read each file, get the array and append it as a row into two dataframes



# In[ ]:




