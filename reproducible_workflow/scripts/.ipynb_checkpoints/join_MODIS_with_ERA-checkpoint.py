import re
import glob
import xarray as xr
import pandas as pd
import sys
import numpy as np
from contextlib import suppress
import faiss
import time 

"""
Script to join the ERA data with the MODIS data. 
See Workflow/ 2. Joining Data
"""

#Deal with filename sorting. Stolen from: https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)





#--------------------------------#
#------Global Parameters---------#
#--------------------------------#


root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/'


    
#These dictionaries describe the local hour of the satellite
local_times = {"aquaDay":"13:30",
               "terraDay":"10:30",
               "terraNight":"22:30",
               "aquaNight":"01:30"
              }
# and are used to load the correct file for dealing with the date-line.
min_hours = {"aquaDay":2,
            "terraDay":-1,
            "aquaNight":-1,
            "terraNight":11}
max_hours = {"aquaDay":24,
            "terraDay":22,
            "aquaNight":13,
            "terraNight":24}


satellite = 'aquaDay'
satellite_folder = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/MODIS'
previous_datestring = None
latitude_bound = 70

IO_path = f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/'





#-------------------------------------------------#
#---------Load any data and declare paths---------#
#-------------------------------------------------#
print ('Pre loading data')

#Time variable ERA data
ERA_folder = root+'processed_data/ERA_timevariable/'
ERA_files = natural_sort(glob.glob(ERA_folder+'*'))

#Time constant ERA data. This is different for v15 and v20 data.
versions = ["v15", "v20"]
ERA_constant_dict = {}
for v in versions:
    f = root +f'processed_data/ERA_timeconstant/ERA_constants_{v}.nc'
    ds = xr.open_dataset(f) #NetCDF file of features which are constant for each gridpoint
    
    #Rename
    name_dict={x:x+f'_{v}' for x in list(ds.keys())}
    ds = ds.rename(name_dict)
    
    ERA_constant_dict[v] = ds
    ds.close()

       

            
#'Bonus' ERA data. This is wetlands and lakes
bonus_root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/BonusClimate/'
wetlands_and_lakes = {'COPERNICUS/':'wetlandf',
                      'CAMA/':      'wetlandf',
                      'ORCHIDEE/':  'wetlandf',
                      'monthlyWetlandAndSeasonalWater_minusRiceAllCorrected_waterConsistent/':'wetlandf',
                      'CL_ECMWFAndJRChistory/':'clake'} #Monthly varying data `cldiff`. Directory:Filename

wetlands_ds = xr.Dataset() #Empty ds
for w in wetlands_and_lakes:
    fname = bonus_root+w+wetlands_and_lakes[w]    
    #Load
    ds_wetland= xr.open_dataset(fname,engine='cfgrib',decode_times=False,backend_kwargs={'indexpath': ''}) 
    
    #Rename the parameter so everything is not cldiff
    ds_wetland = ds_wetland.cldiff.rename(f'cldiff_{w}') #This is now a dataarray

    #Fix the time to be an integer
    ds_wetland['time'] = np.arange(1,12+1) #i.e. what month it it?
    
    #Output
    wetlands_ds[w] = ds_wetland    
    

    
#------------------------#
#------Functions---------#
#------------------------#


def get_ERA_hour(ERA_month,t,v15,v20,WetlandsAndLakesMonth,bounds):
    
    """
    Extract an hour of ERA data
    """
    
    #Filter month of ERA data to an hour
    time_filter = (ERA_month.time == t)
    ERA_hour = ERA_month.where(time_filter,drop=True)
    
    
    #Join on the constant data V15 and v20, and the wetlands data, first setting the time coordinate
    v15 = v15.assign_coords({"time": (((ERA_hour.time)))}) 
    v20 = v20.assign_coords({"time": (((ERA_hour.time)))}) 
    WetlandsAndLakesMonth = WetlandsAndLakesMonth.assign_coords({"time": (((ERA_hour.time)))})
    
    ERA_hour = xr.merge([ERA_hour,v15,v20, WetlandsAndLakesMonth]).load() #Explicitly load 
    
    
    #And covert longitude to long1
    ERA_hour = ERA_hour.assign_coords({"longitude": (((ERA_hour.longitude + 180) % 360) - 180)})
    
    
        
    # Also filter by latitude/longtiude
    longitude_filter = (ERA_hour.longitude > bounds['longitude_min']) & (ERA_hour.longitude < bounds['longitude_max'])
    latitude_filter =  (ERA_hour.latitude > bounds['latitude_min']) & (ERA_hour.latitude < bounds['latitude_max'])
    return ERA_hour.where(longitude_filter & latitude_filter,drop=True)
  
    
    

    #return ERA_hour



def select_correct_MODIS_file(t):
    
    """We have to be careful with the dateline. This function
       figures out which MODIS file to load."""

    
    #Get the hour
    utc_hour = t.hour
    
    
    #Due to crossing of the datetime, some times will be saved different date
    if utc_hour < min_hours[satellite]:
        file_date = t  - np.timedelta64(1,'D')
    elif utc_hour > max_hours[satellite]:
        file_date = t  + np.timedelta64(1,'D')
    else:
        file_date = t
        
    #Create a string which will be used to open file
    y = pd.to_datetime(file_date).year
    m = pd.to_datetime(file_date).month
    d = pd.to_datetime(file_date).day
    date_string = f'{y}-{m:02}-{d:02}'
    
    return date_string



def load_MODIS_file(date_string):
    
    """
    Load a day of MODIS data, apply some filters and corrections
    """
    
    #Open that file
    MODIS_data = xr.open_dataarray(f'{satellite_folder}/{satellite}_errorGTE03K_04km_{date_string}.tif',engine="rasterio")

    #Make some edits to file
    MODIS_data = MODIS_data.rename({'x':'longitude','y':'latitude'})

    #Filter by latitude bound
    space_filter = np.expand_dims(np.abs(MODIS_data.latitude) < latitude_bound,axis=(0,-1))
    mask = np.logical_and(np.isfinite(MODIS_data),space_filter) #make it a 2d mask
    MODIS_data = MODIS_data.where(mask,drop=True)

    #Convert local satellite time to UTC and round to nearest hour
    time_delta = pd.to_timedelta(MODIS_data.longitude.data/15,unit='H') 
    time_UTC = (pd.to_datetime([date_string + " " + local_times[satellite]]*time_delta.shape[0]) - time_delta).round('H')

    return MODIS_data,time_UTC


def haver(lat1_deg,lon1_deg,lat2_deg,lon2_deg):
    
    """
    Given coordinates of two points IN DEGREES calculate the haversine distance
    """
    
    #Convert degrees to radians
    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)


    #...and the calculation
    delta_lat = lat1 -lat2
    delta_lon = lon1 -lon2
    Re = 6371 #km
    Z = np.sin(delta_lat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(delta_lon/2)**2
    H = 2*Re*np.arcsin(np.sqrt(Z)) #Haversine distance in km
    return H



def faiss_knn(database,query):
    
    """
    Use faiss library (https://github.com/facebookresearch/faiss) for fass k-nearest neighbours on GPU
    
    Note that the nearness is an L2 (squared) norm on the lat/long coordinates, rather than a haversine metric
    """
    
    #Database
    xb = database[["latitude", "longitude"]].to_numpy().astype('float32')
    xb = xb.copy(order='C') #C-contigious
    
    #Query
    xq = query[["latitude", "longitude"]].to_numpy().astype('float32') 
    xq = xq.copy(order='C')
    
    #Create index
    d = 2                            # dimension
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(d) #index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat) # make it into a gpu index
    gpu_index_flat.add(xb)  
    
    #Search
    k = 1                          # we want to see 1 nearest neighbors
    distances, indices = gpu_index_flat.search(xq, k)
        
   
    df = query.reset_index().join(database.iloc[indices.flatten()].reset_index(), lsuffix='_MODIS',rsuffix='_ERA')
    df['L2_distance'] = distances
    df['H_distance'] = haver(df['latitude_MODIS'],df['longitude_MODIS'],df['latitude_ERA'],df['longitude_ERA']) #Haversine distance
    
    #Filter out any large distances
    tolerance = 50 #km
    df_filtered = df.query('H_distance < %.9f' % tolerance)


    #Group it. Each ERA point has a bunch of MODIS points. Group and average
    df_grouped = df_filtered.groupby(['latitude_ERA','longitude_ERA'],as_index=False).mean()

    
    return df_grouped
    




selection_index = 28 #Use if you dont want to run for all the ERA files e.g. script gets killed after X months
selected_ERA_files = ERA_files[selection_index:] 
print(selected_ERA_files)
counter = selection_index  

print ('Iterating over all months:')
for f in selected_ERA_files:
    
    #Load a month of ERA data
    print ('Loading ERA month:', f)
    ERA_month = xr.open_dataset(f,engine='cfgrib',backend_kwargs={'indexpath': ''})
    
    #Get all times in that month of data, hourly grain
    timestamps = pd.to_datetime(ERA_month.time) 
    
    
    #Load the wetland bonus data for that month
    month = np.unique(timestamps.month)[0] #There should only be one value, an integer in range 1-12
    wetlands_time_filter = (wetlands_ds.time == month)
    wetlands_month = wetlands_ds.where(wetlands_time_filter,drop=True) #This is a field at a single time.
    
        
    #Empty. We will append the resulting dfs here
    dfs = []
    for t in timestamps:
        
        print(t)
        #Get an hour of MODIS data
        date_string = select_correct_MODIS_file(t) #For this datetime, which MODIS file should be opened? 
        if date_string == '2017-12-31': continue #skip we dont have this day
            
        if date_string != previous_datestring:
            #We need to open a new file. 
            with suppress(NameError):MODIS_data.close() #First close the old one explicitly. Exception handles case where MODIS_data not yet defined
            MODIS_data,time_UTC = load_MODIS_file(date_string)
            previous_datestring = date_string
    
        #Filter to only select the hour of data we want
        time_filter = np.expand_dims(time_UTC == t,axis=(0,1))
        mask = np.logical_and(np.isfinite(MODIS_data),time_filter)
        MODIS_hour = MODIS_data.where(mask,drop=True).load() 
        MODIS_df = MODIS_hour.to_dataframe(name='MODIS_LST').reset_index().dropna() #Make everything a pandas df to pass into faiss_knn. Unnecessary step?

    
        #Spatial bounds
        #Get the limits of the MODIS box. We will use this to filter the ERA data for faster matching
        #i.e. when looking for matches, dont need to look over the whole Earth, just a strip
        delta = 1.0 # Enveloping box
        bounds = {"latitude_min" :MODIS_df.latitude.min()-delta,
                  "latitude_max" :MODIS_df.latitude.max()+delta,
                  "longitude_min":MODIS_df.longitude.min()-delta,
                  "longitude_max":MODIS_df.longitude.max()+delta
          }
    
    
    
        #Get an hour of ERA data
        ERA_hour = get_ERA_hour(ERA_month,t, ERA_constant_dict['v15'], ERA_constant_dict['v20'],wetlands_month,bounds) #Get an hour of ERA data
        ERA_df = ERA_hour.to_dataframe().reset_index() #Make it a df

        #Find matches in space
        df_matched = faiss_knn(ERA_df,MODIS_df) #Match reduced gaussian grid to MODIS
        df_matched['time'] = t            
        df_matched = df_matched.drop(['index_MODIS', 'band','spatial_ref','index_ERA','values','number','surface','depthBelowLandLayer'], axis=1) #get rid of all these columns that we dont need
        dfs.append(df_matched)
        
      
    
        #Explicitly deallocate
        ERA_hour.close()
        MODIS_hour.close()
        
    #At the end of every month, do some IO
    #Pkl is likely suboptimial here. Need to update to e.g. parquet, HDF, etc.

    df = pd.concat(dfs)
    fname = f'V2matched_{counter}.pkl'
    print ("Writing to disk:", IO_path+fname)
    df.to_pickle(IO_path+fname)
        
        
    counter += 1
    #Deallocate
    ERA_month.close()
        
        
        


