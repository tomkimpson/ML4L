import xarray as xr
import rioxarray
import pandas as pd
import numpy as np
import faiss
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
from datetime import timedelta, date



"""
Script join monthly ERA .grib data with daily MODIS .tiff data
We join in both time (hourly grain) and space.
See X
"""



###---GLOBAL VARIABLES---###


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


#Path to MODIS data
satellite_folder = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/MODIS'

#Path to ERA data
era_folder = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/'


#Where do you want the output files to go?
IO_path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/monthly_joined_ERA_MODIS/'

#Some parameters - can be changed by user.
satellite='aquaDay'
latitude_bound=70
verbose = False

###------END------###



###---FUNCTIONS---###

#Stolen from https://stackoverflow.com/questions/5980042/how-to-implement-the-verbose-or-v-option-into-a-script
if verbose:
    def verboseprint(*args):
        # Print each argument separately so caller doesn't need to
        # stuff everything to be printed into a single string
        for arg in args:
            print (arg,)
        print
else:   
    verboseprint = lambda *a: None      # do-nothing function


def fetch_ERA_month(year_month):
    
    """Load a month of ERA data from multiple sources and filter to just get land values"""
    
    print ('Loading a month of ERA data. These are large files so could take some time')
    
    #All different data sources
    fskin = era_folder+'ERA_skin/sfc_skin_unstructured_'+year_month+'.grib'
    fsfc =  era_folder+'ERA_sfc/sfc_unstructured_'+year_month+'.grib'
    fskt =  era_folder+'ERA_skt/skt_unstructured_'+year_month+'.grib'

    #Load month of data from different sources
    ds_skin = xr.open_dataset(fskin,engine='cfgrib',filter_by_keys={'typeOfLevel': 'surface'},backend_kwargs={'indexpath': ''})
    ds_sfc = xr.open_dataset(fsfc,engine='cfgrib',filter_by_keys={'typeOfLevel': 'surface'},backend_kwargs={'indexpath': ''})
    ds_skt = xr.open_dataset(fskt,engine='cfgrib',filter_by_keys={'typeOfLevel': 'surface'},backend_kwargs={'indexpath': ''})

    #...and merge it into one
    ERA = xr.merge([ds_skin, ds_sfc,ds_skt])
    
    #Filter to just get land values
    land_filter = (ERA.lsm > 0.5)
    ERA_land = ERA.where(land_filter,drop=True)

    
    #Relabel longitude coordinate to be consistent with MODIS
    ERA_land = ERA_land.assign_coords({"longitude": (((ERA_land.longitude + 180) % 360) - 180)})

    return ERA_land







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



def process_MODIS_file(sat_xr,date_string,latitude_bound): 
    
    """
    Rename some columns, apply latitude bounds and calculate absolute time from local solar time
    """
    
    # Rename spatial dimensions
    sat_xr = sat_xr.rename({'x':'longitude','y':'latitude'})
    
    
    #Filter by latitude
    space_filter = np.expand_dims(np.abs(sat_xr.latitude) < latitude_bound,axis=(0,-1))
    mask = np.logical_and(np.isfinite(sat_xr),space_filter) #make it a 2d mask

    sat_xr = sat_xr.where(mask,drop=True)
    
    #Create time delta to change local to UTC
    time_delta = pd.to_timedelta(sat_xr.longitude.data/15,unit='H') 
    
    #Convert local satellite time to UTC and round to nearest hour
    time = (pd.to_datetime([date_string + " " + local_times[satellite]]*time_delta.shape[0]) - time_delta).round('H')
    
    
    return sat_xr.where(mask,drop=True), time




def snapshot_MODIS(t,MODIS_data,MODIS_time,previous_datestring):
    
    """
    Select an hourly snapshot of MODIS data
    """
    
    verboseprint('Getting an hourly snapshot of MODIS data')
    
    date_string = select_correct_MODIS_file(t)
    
    if date_string != previous_datestring:
        #We need to open a new file. 
        #First close the old one explicitly
        try:
            MODIS_data.close()
        except:
            pass
        
        #Now open a new file, and update the datestring
        print('Opening new file')
        MODIS_data = xr.open_dataarray(f'{satellite_folder}/{satellite}_errorGTE03K_04km_{date_string}.tif',engine="rasterio")
        previous_datestring=date_string
            
        #Make some corrections and calculations
        MODIS_data,MODIS_time = process_MODIS_file(MODIS_data,date_string,latitude_bound)      
   

    #Select the correct hour of MODIS data
    time_filter = np.expand_dims(MODIS_time == t,axis=(0,1))
    
    # Make this 1d time filter a 2d mask
    mask = np.logical_and(np.isfinite(MODIS_data),time_filter)
    
    #Apply mask to data array
    MODIS_data_snapshot= MODIS_data.where(mask,drop=True) 
    
    return MODIS_data_snapshot,MODIS_data,MODIS_time,date_string #hour of data, day of data, times in UTC, date_string
    
    
    
def snapshot_ERA(t,ERA_land,MODIS_data_snapshot):

    
    """
    Select an hourly snapshot of ERA data, spatially filtered
    """

    verboseprint('Getting an hourly snapshot of ERA data')

    
    #Filter month of ERA land data to an hour 
    time_filter = (ERA_land.time == t)
    ERA_land_snapshot = ERA_land.where(time_filter,drop=True)
    
    
    
    
    #Filter spatially - MODIS is only a strip of data, we dont need the whole Earth surface
    delta = 1.0 #some leeway
    bounds = {"latitude_min"  :MODIS_data_snapshot.latitude.data.min()-delta,
              "latitude_max"  :MODIS_data_snapshot.latitude.data.max()+delta,
               "longitude_min":MODIS_data_snapshot.longitude.data.min()-delta,
               "longitude_max":MODIS_data_snapshot.longitude.data.max()+delta
              }
    

    longitude_filter = (ERA_land_snapshot.longitude > bounds['longitude_min']) & (ERA_land_snapshot.longitude < bounds['longitude_max'])
    latitude_filter =  (ERA_land_snapshot.latitude > bounds['latitude_min']) & (ERA_land_snapshot.latitude < bounds['latitude_max'])
    ERA_land_snapshot = ERA_land_snapshot.where(longitude_filter & latitude_filter,drop=True)
    

    return ERA_land_snapshot

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
    

    #Combine into a single df with all data
    df = query.reset_index().join(database.iloc[indices.flatten()].reset_index(), lsuffix='_MODIS',rsuffix='_ERA')
    df['L2_distance'] = distances
    df['MODIS_idx'] = indices
    df['H_distance'] = haver(df['latitude_MODIS'],df['longitude_MODIS'],df['latitude_ERA'],df['longitude_ERA']) #Haversine distance
    
    #Filter out any large distances
    tolerance = 50 #km
    df_filtered = df.query('H_distance < %.9f' % tolerance)


    #Group it. Each ERA point has a bunch of MODIS points. Group and average
    df_grouped = df_filtered.groupby(['latitude_ERA','longitude_ERA'],as_index=False).mean()

    
    return df_grouped







###------END------###







###------MAIN------###



#Parameters
dates = pd.date_range(start="2018-01-01",end="2020-01-01",freq='MS') #Iterate over this date range



#Declarations
MODIS_data_snapshot,MODIS_data,MODIS_time,previous_datestring = (None,)*4 #Declare some empty variables

for dt in dates:
    print (dt)
    year_month = dt.strftime("%Y_%m")
    ERA_land = fetch_ERA_month(year_month) #This is a month of ERA data
    
    timestamps = pd.to_datetime(ERA_land.time) # hourly grain
    
    if dt == dates[0]:
        timestamps = timestamps[3:] #We dont have data for Dec 2017. Can't do 0,1,2. Note this correction is specific to AquaDay
        
        
    dfs = []
    for t in timestamps:
        print(t)
        
        #Populate previously empty MODIS variables. 
            #MODIS_data_snapshot = hour of MODIS data
            #MODIS_data = day of MODIS data
            #MODIS_time = UTC times of MODIS_data
            #previous_datestring = used to determine whether we need to open a new file, or use the one we already have
        MODIS_data_snapshot,MODIS_data,MODIS_time,previous_datestring = snapshot_MODIS(t,MODIS_data,MODIS_time,previous_datestring)


        #And get hour of ERA data
        ERA_data_snapshot = snapshot_ERA(t,ERA_land,MODIS_data_snapshot)
        
        
        #Make everything a pandas df to pass into faiss_knn. Unnecessary step?
        ERA_df = ERA_data_snapshot.to_dataframe().reset_index().dropna()
        MODIS_df = MODIS_data_snapshot.to_dataframe().reset_index().dropna()
        
        
        verboseprint(ERA_df)
        verboseprint(MODIS_df)
        
        df_matched = faiss_knn(ERA_df,MODIS_df)
        df_matched['time'] = t
        dfs.append(df_matched)
        
        
    #At the end of every month, do some IO
    df = pd.concat(dfs)
    fname = f'matched_{dt}.pkl'
    print ("Writing to disk:", IO_path+fname)
    df.to_pickle(IO_path+fname)

###------END------###






    
# def faiss_knn_swp(database,query):
    
#     """
#     Use faiss library (https://github.com/facebookresearch/faiss) for fass k-nearest neighbours on GPU
    
#     Note that the nearness is an L2 (squared) norm on the lat/long coordinates, rather than a haversine metric
#     """
    
#     #Database
#     xb = database[["latitude", "longitude"]].to_numpy().astype('float32')
#     xb = xb.copy(order='C') #C-contigious
    
#     #Query
#     xq = query[["latitude", "longitude"]].to_numpy().astype('float32') 
#     xq = xq.copy(order='C')
    
#     #Create index
#     d = 2                            # dimension
#     res = faiss.StandardGpuResources()
#     index_flat = faiss.IndexFlatL2(d) #index
#     gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat) # make it into a gpu index
#     gpu_index_flat.add(xb)  
    
#     #Search
#     k = 1                          # we want to see 1 nearest neighbors
#     distances, indices = gpu_index_flat.search(xq, k)
    

#     #Combine into a single df with all data
#     df = query.reset_index().join(database.iloc[indices.flatten()].reset_index(), lsuffix='_ERA',rsuffix='_MODIS')
#     df['L2_distance'] = distances
#     df['MODIS_idx'] = indices
#     df['H_distance'] = haver(df['latitude_MODIS'],df['longitude_MODIS'],df['latitude_ERA'],df['longitude_ERA']) #Haversine distance
    
    
#     #Filter out any large distances
#     tolerance = 50 #km
#     df_filtered = df.query('H_distance < %.9f' % tolerance)


  

    
#     return df_filtered













 