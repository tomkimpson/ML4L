import pandas as pd
import xarray as xr
import numpy as np
import sys
from config import *
import glob
import rioxarray as rxr


def process_x_df(df):
    #Convert to long1 and set the index
    df['latitude'] = np.round(df.index.get_level_values('latitude').values,3)
    df['longitude'] = np.round((df.index.get_level_values('longitude').values +180.0) %360.0 - 180.0,3)

 
    df = df.set_index(['latitude', 'longitude'], drop=True) #reindex
    
    drop_columns = ['valid_time','number', 'step', 'surface']
    df = df.drop(drop_columns, axis=1) #get rid of columns we don't need

    return df.dropna()


def process_y_df(df):

    #Reindex dfy via a linear shift
    #---ATTENTION---!> We add a linear shift of 0.0250 such that the coordinates match between the X and Y data
    # We need to clarify the proper way to deal with this. Perhaps some interpolation method?
    df['latitude'] = np.round(df.index.get_level_values('y').values - 0.0250,3)
    df['longitude'] = np.round(df.index.get_level_values('x').values - 0.0250,3)
    


    df = df.set_index(['latitude', 'longitude'], drop=True)



    selected_y_columns = ['LST_Day_CMG'] #only use these columns, drop the others
    df = df[selected_y_columns]




    return df.dropna()





#Load the entire X data, lasily
cds_xarray = xr.open_dataset(data_root+"CDS.grib",engine='cfgrib')

print ('opened x data OK')

#Path to individual Y data
all_MODIS_files = glob.glob(data_root+'e4ftl01.cr.usgs.gov/MOLT/MOD11C3.006/**/*.hdf')

dfs = []
counter = 0
for f in (all_MODIS_files):
    print(f)
    print('Counter = ', counter)
   
    #Load Y data from HDF
    modis_xarray= rxr.open_rasterio(f,masked=True)
    datastamp = modis_xarray.attrs['RANGEBEGINNINGDATE']
    modis_df = modis_xarray.to_dataframe() #everything as a df
    modis_df['time'] = np.datetime64(datastamp)
    modis_df = modis_df[['LST_Day_CMG', 'time']].dropna() 
    
    #Get corresponding X data from CDS
    
    cds_i = cds_xarray.sel(time=np.datetime64(datastamp)).to_dataframe()
    
    #Clean the data
    df_x_clean = process_x_df(cds_i)
    df_y_clean = process_y_df(modis_df)
    

    
    #Merge and reindex
    df_merged = pd.merge(df_x_clean,df_y_clean,how='inner',left_index=True, right_index=True)
    df_merged = df_merged.set_index(['time'], append=True) #turn time into index

    #Drop duplicates. OPTIONAL
    df_i = df_merged.drop_duplicates()
    
    #IO
    dfs.append(df_i)
    counter += 1



#This is probably poor practice just doing one big IO  
df_clean = pd.concat(dfs)
df_clean.to_pickle(clean_data)
#df_clean.to_hdf(data_root+'largecleaned.hdf','mycleandata',mode='w')






