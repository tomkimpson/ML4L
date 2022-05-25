import re
import glob
import xarray as xr
import pandas as pd
import sys
import numpy as np
from contextlib import suppress
import faiss

#Deal with filename sorting. Stolen from: https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)





#--------------------------------#
#------Global Parameters---------#
#--------------------------------#


root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/'



#Get reduced Gaussian Grid, |lat| < 70
ds = xr.open_dataset(root +f'processed_data/ERA_timeconstant/ERA_constants_v15.nc') # NetCDF file on the grid we are using
ds = ds.assign_coords({"longitude": (((ds.longitude + 180) % 360) - 180)})          # Change longitude to match MODIS
index_file = ds.to_dataframe().reset_index()                                        # Make it a df
index_file = index_file[['latitude','longitude']]                                   # We only care about these columns
index_file = index_file.query("abs(latitude) <= 70.0")                              # And we don't want poles

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

IO_path = f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/tmp_index/'

#------------------------#
#------Functions---------#
#------------------------#
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
    
    #Select the ERA coordinates which are matched to a MODIS data point
    selected_ERA = database.iloc[indices.flatten()]
    
    print('ERA')
    print(selected_ERA)
    
    #Combine into a single df
    df = query.reset_index().join(selected_ERA.reset_index(), lsuffix='_MODIS',rsuffix='_ERA')
    df = df.set_index(selected_ERA.index) #Set the index
    df['H_distance'] = haver(df['latitude_MODIS'],df['longitude_MODIS'],df['latitude_ERA'],df['longitude_ERA']) #Haversine distance
    
    print('Joined')
    print(df)
    
    #Filter out any large distances.
    #Doesnt usually happen since MODIS grid points at higher resolution, but just a safety catch
    tolerance = 50 #km
    df_filtered = df.query('H_distance < %.9f' % tolerance)
   
    #Group it. Each ERA point has a bunch of MODIS points. Group and average
    #df_grouped = df_filtered.groupby(df_filtered.index).mean()
    df_grouped = df_filtered.groupby(['latitude_ERA','longitude_ERA','time_UTC'],as_index=False).mean()

    print('Grouped')
    print (df_grouped)
    
    return df_grouped[['latitude_ERA', 'longitude_ERA','time_UTC', 'MODIS_LST']] #we only want these columns




MODIS_files = sorted(glob.glob(f'{satellite_folder}/{satellite}_errorGTE03K_04km_*.tif'))
import time
start_time = time.time()
for f in MODIS_files:
    print(f)

    MODIS_data = xr.open_dataarray(f,engine="rasterio")
    date_string = f.split('04km_')[-1].split('.')[0] #Hacky
    
    #Make some edits to file
    MODIS_data = MODIS_data.rename({'x':'longitude','y':'latitude'})

    #Filter by latitude bound
    space_filter = np.expand_dims(np.abs(MODIS_data.latitude) < latitude_bound,axis=(0,-1))
    mask = np.logical_and(np.isfinite(MODIS_data),space_filter) #make it a 2d mask
    MODIS_data = MODIS_data.where(mask,drop=True)

    #Make it a pandas df
    MODIS_df = MODIS_data.to_dataframe(name='MODIS_LST').reset_index().dropna() #Make everything a pandas df to pass into faiss_knn. Unnecessary step?



    MODIS_df['time_delta'] = pd.to_timedelta(MODIS_df.longitude/15,unit='H') 
    MODIS_df['reference_time'] = date_string + " " + local_times[satellite]
    MODIS_df['time_UTC'] = (pd.to_datetime(MODIS_df.reference_time) - MODIS_df.time_delta).round('H')


    print(MODIS_df)
    
    print('Find matched')        
#     #Find matches in space
    df_matched = faiss_knn(index_file,MODIS_df)
#     df_matched['time'] = t
#     dfs.append(df_matched)



    #Do some IO
    print ('Doing IO')
    fname = f'matched_index_{date_string}.pkl'
    df_matched.to_pickle(IO_path+fname)
    print('saved to', IO_path+fname)

    print('End time =', time.time() - start_time)
    sys.exit()



# timestamps = pd.date_range(start="2018-01-01",end="2020-12-31",freq='H') #Iterate over this date range
# current_month = 1
# dfs = []
# for t in timestamps:
#     print (t)
    
#     if t.month != current_month:
#         #Do some IO
#         print ('Doing IO')
#         fname = f'matched_index_{t.year}-{t.month:02}.pkl'
#         df = pd.concat(dfs)
#         df.to_pickle(IO_path+fname)
#         dfs = []
#         current_month = t.month
    

#     #Get an hour of MODIS data
#     date_string = select_correct_MODIS_file(t) #For this datetime, which MODIS file should be opened? 
#     if date_string == '2017-12-31': continue #skip we dont have this day
            
#     if date_string != previous_datestring:
#         #We need to open a new file. 
#         with suppress(NameError):MODIS_data.close() #First close the old one explicitly. Exception handles case where MODIS_data not yet defined
#         MODIS_data,time_UTC = load_MODIS_file(date_string)
#         previous_datestring = date_string
    
#     #Filter to only select the hour of data we want
#     time_filter = np.expand_dims(time_UTC == t,axis=(0,1))
#     mask = np.logical_and(np.isfinite(MODIS_data),time_filter)
#     MODIS_hour = MODIS_data.where(mask,drop=True).load() 
#     MODIS_df = MODIS_hour.to_dataframe(name='MODIS_LST').reset_index().dropna() #Make everything a pandas df to pass into faiss_knn. Unnecessary step?

    
#     #Find matches in space
#     df_matched = faiss_knn(index_file,MODIS_df)
#     df_matched['time'] = t
#     dfs.append(df_matched)

    
#     #Deallocate
#     MODIS_hour.close()

    
# #IO once
# p

        


