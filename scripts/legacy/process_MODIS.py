import rioxarray as rxr
import glob
import pandas as pd
import numpy as np
from config import *

all_files = glob.glob(data_root+'e4ftl01.cr.usgs.gov/MOLT/MOD11C3.006/**/*.hdf')

dfs = []
for f in (all_files):
    print (f)
    modis_xarray= rxr.open_rasterio(f,masked=True)
    datastamp = modis_xarray.attrs['RANGEBEGINNINGDATE']
    modis_df = modis_xarray.to_dataframe() #everything as a df
    modis_df['time'] = np.datetime64(datastamp)
    print ('datastamp')
    modis_df = modis_df[['LST_Day_CMG', 'time']] 

    dfs.append(modis_df)

df_modis = pd.concat(dfs)
df_modis.to_pickle(data_root+"modis.pkl")


