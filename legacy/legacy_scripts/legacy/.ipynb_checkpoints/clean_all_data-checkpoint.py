import pandas as pd
import xarray as xr
import numpy as np
import sys
from config import *

def index_level_dtypes(df):
    return [f"{df.index.names[i]}: {df.index.get_level_values(n).dtype}"
            for i, n in enumerate(df.index.names)]

def process_x_df(df):
    
    #Convert to long1 and set the index
    df['latitude'] = np.round(df.index.get_level_values('latitude').values,3)
    df['longitude'] = np.round((df.index.get_level_values('longitude').values +180.0) %360.0 - 180.0,3)
    df['time'] = df.index.get_level_values('time').values

    print ('X time:')
    print (np.unique(df['time']))

    df = df.set_index(['latitude', 'longitude','time'], drop=True)

    return df.dropna()


def process_y_df(df):

    #Reindex dfy via a linear shift
    #---ATTENTION---!> We add a linear shift of 0.0250 such that the coordinates match between the X and Y data
    # We need to clarify the proper way to deal with this. Perhaps some interpolation method?
    df['latitude'] = np.round(df.index.get_level_values('y').values - 0.0250,3)
    df['longitude'] = np.round(df.index.get_level_values('x').values - 0.0250,3)
    
    print ('Y time:')
    print (np.unique(df['time']))

    df = df.set_index(['latitude', 'longitude','time'], drop=True)



    selected_y_columns = ['LST_Day_CMG'] #only use these columns, drop the others
    df = df[selected_y_columns]


    return df.dropna()





#Load the data
cds_xarray = xr.open_dataset(data_root+"xdata.nc")
df_x = cds_xarray.to_dataframe()
df_y = pd.read_pickle(data_root+'modis.pkl')


#Process and clean the data
df_x_clean = process_x_df(df_x)
df_y_clean = process_y_df(df_y)


#Process the X data
df_merged = pd.merge(df_x_clean,df_y_clean,how='inner',left_index=True, right_index=True)
df_merged.to_pickle(data_root+"df_clean.pkl")

print (df_merged)














