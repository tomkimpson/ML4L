import xarray as xr
import rioxarray
import pandas as pd
import numpy as np
import faiss
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
from datetime import timedelta, date


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
satellite_folder = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/MODIS/'

#Path to ERA data
era_folder = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw'


def get_satellite_slice(date : str,
                        utc_hour : int,
                        satellite : str,
                        latitude_bound = None #Recommend only using |lat| < 70 degrees
                        ):
    
    """Function to load hourly slice of MODIS data from Mat Chantry.
       Some naming changes from original, logic mostly unchanged.
       We now drop coordinates that are filtered out"""
    
    #Due to crossing of the datetime, some times will be saved different date
    if utc_hour < min_hours[satellite]:
        file_date = str((np.datetime64(date) - np.timedelta64(1,'D')))
    elif utc_hour > max_hours[satellite]:
        file_date = str((np.datetime64(date) + np.timedelta64(1,'D')))
    else:
        file_date = date
        
    # Open .tif 
    #sat_xr = xr.open_rasterio(f'{satellite_folder}/{satellite}_errorGTE03K_04km_{file_date}.tif')
    sat_xr = xr.open_dataarray(f'{satellite_folder}/{satellite}_errorGTE03K_04km_{file_date}.tif',engine="rasterio")

    # Rename spatial dimensions
    sat_xr = sat_xr.rename({'x':'longitude','y':'latitude'})
    
    #Create time delta to change local to UTC
    time_delta = pd.to_timedelta(sat_xr.longitude.data/15,unit='H') 
    
    #Convert local satellite time to UTC and round to nearest hour
    time = (pd.to_datetime([file_date + " " + local_times[satellite]]*time_delta.shape[0]) - time_delta).round('H')
    
    #What date/time does the user want?
    target_time = np.datetime64(f'{date} {utc_hour:02}:00:00')
        
    #Is this target time in this data array?
    time_filter = np.expand_dims(time == target_time,axis=(0,1))
    
    # Make this 1d time filter a 2d mask
    mask = np.logical_and(np.isfinite(sat_xr),time_filter)
    
    # Also filter by latitude
    space_filter = np.expand_dims(np.abs(sat_xr.latitude) < latitude_bound,axis=(0,-1))
    
    #...and add this condition to the mask
    mask = np.logical_and(mask,space_filter)
    
    #Check we have some true values in our mask
    if mask.sum() == 0:
        print('There is no appropriate data')
        return 0
      
    #Apply mask to data array
    sat_xr_filtered= sat_xr.where(mask,drop=True).load() 
     
    #Close
    sat_xr.close()
    sat_xr = None
    
    
   
    return sat_xr_filtered[0,::-1,:]





def get_era_data(date : str,
                 utc_hour : str,
                 field : str,
                 bounds : dict,
                 source: str):

    """Function to load hourly slice of ERA data 
       Additional filtering by max/min longitude, read from matching MODIS file
    """
        
    #Load the data
    month = '_'.join(date.split('-')[:-1])
    
    if source == 'ERA_skin':
        name = '_skin_'
    if source == 'ERA_sfc':
        name = '_'
        
    utc_0hour = f'{utc_hour:02}'
    fname = f'{era_folder}/{source}/NetCDF/{date}T{utc_0hour}:00:00.000000000.nc'
    ds_era = xr.open_dataset(fname)
    
        
    #Grab correct field
    if field is not None:
        da = ds_era[field]
    else:
        da = ds_era
        
    
     # Also filter by latitude/longtiude
    longitude_filter = (da.longitude > bounds['longitude_min']) & (da.longitude < bounds['longitude_max'])
    latitude_filter =  (da.latitude > bounds['latitude_min']) & (da.latitude < bounds['latitude_max'])
    
  
    da_filtered = da.where(longitude_filter & latitude_filter,drop=True)

    #Explictley close the file
    ds_era.close()
    
    return da_filtered
    



    
def filter_out_sea(ds : xr.Dataset):
    
    """
    Use lsm variable to filter out ocean values, retaining only land values
    Returns a pandas dataframe
    """
    
    df = ds.to_dataframe().reset_index()
    df_land = df.loc[df['lsm'] > 0.5]
    
    
    return df_land
    
    
    

    
    
def find_closest_match_sklearn(MODIS_df : pd.DataFrame, 
                               ERA_df : pd.DataFrame,
                               tolerance : float):
    

    
    #We only want non null MODIS values
    MODIS_df = MODIS_df.dropna()
    

    
    #Construct NN     
    NN = NearestNeighbors(n_neighbors=1, metric='haversine') #algorithm = balltree, kdtree or brutie force


    NN.fit(np.deg2rad(MODIS_df[['latitude', 'longitude']].values))


    query_lats = ERA_df['latitude'].astype(np.float64)
    query_lons = ERA_df['longitude'].astype(np.float64)
    X = np.deg2rad(np.c_[query_lats, query_lons])
    distances, indices = NN.kneighbors(X, return_distance=True)


    r_km = 6371 # multiplier to convert to km (from unit distance)
    distances = distances*r_km


    
    df_combined = ERA_df.reset_index().join(MODIS_df.iloc[indices.flatten()].reset_index(), lsuffix='_ERA',rsuffix='_MODIS')
    df_combined['distance'] = distances
    df_combined['MODIS_idx'] = indices

    #Filter and surface selected columns
    df_combined_matches = df_combined.query('distance < %.9f' % tolerance)

    
    
    return df_combined_matches.dropna().reset_index()   
    #return df_combined.reset_index()
    
    
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



    
def faiss_knn_swp(database,query):
    
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
    df = query.reset_index().join(database.iloc[indices.flatten()].reset_index(), lsuffix='_ERA',rsuffix='_MODIS')
    df['L2_distance'] = distances
    df['MODIS_idx'] = indices
    df['H_distance'] = haver(df['latitude_MODIS'],df['longitude_MODIS'],df['latitude_ERA'],df['longitude_ERA']) #Haversine distance
    
    
    #Filter out any large distances
    tolerance = 50 #km
    df_filtered = df.query('H_distance < %.9f' % tolerance)


  

    
    return df_filtered












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

    
    


def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)
        
        
def pipeline(date,utc_hour,satellite,latitude_bound,ERA_fields,tolerance,method):

    # MODIS data
    MODIS = get_satellite_slice(date,utc_hour,satellite,latitude_bound)
    
    if not isinstance(MODIS,xr.DataArray):
        return pd.DataFrame() #If there is no appropriate data it just returns a 0. We then create an empty pd df so no matches will be found
    
    #Make MODIS ds a df
    MODIS_df = MODIS.to_dataframe(name='MODIS_LST').reset_index()
    
    #ERA data
    delta = 1.0 #Enveloping box.
    bounds = {"latitude_min" :MODIS.latitude.data.min()-delta,
          "latitude_max" :MODIS.latitude.data.max()+delta,
          "longitude_min":MODIS.longitude.data.min()-delta,
          "longitude_max":MODIS.longitude.data.max()+delta
          }
    
    ERA_sfc = get_era_data(date, utc_hour, field=ERA_fields,bounds=bounds,source='ERA_sfc')
    ERA_skin = get_era_data(date, utc_hour, field=ERA_fields,bounds=bounds,source='ERA_skin')
    ERA = xr.merge([ERA_sfc, ERA_skin])
    
    ERA_df_land = filter_out_sea(ERA) #just the land values
    
    if ERA_df_land.empty:
        return pd.DataFrame()
    

    #Explicitly deallocate everything
    MODIS.close()
    MODIS = None
    ERA_sfc.close()
    ERA_skin.close()
    ERA_sfc = None
    ERA_skin = None
    
    
    #Combine
    if method == 'sklearn':
        df_sk = find_closest_match_sklearn(MODIS_df,ERA_df_land,tolerance)
    if method == 'faiss':
        df_sk = faiss_knn(ERA_df_land,MODIS_df.dropna())
    if method == 'faiss_swp':
        df_sk = faiss_knn_swp(MODIS_df.dropna(),ERA_df_land)
    
    return df_sk       
        





#Parameters
#start_date = date(2018, 1, 2)
start_date = date(2020, 8, 20)
end_date   = date(2020, 12, 30)
dates = daterange(start_date,end_date)
hours = np.arange(0,24)



satellite='aquaDay'
latitude_bound=70
ERA_fields = None 
tolerance = 50 #km
method = 'faiss_swp'

#Path for where to ouput saved files
IO_path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/joined_ML_data_faiss_swp/'

print ("-------------------------")
print ('Starting MODIS-ERA joining with the following parameters')
print ("Satellite", satellite)
print ("Date range:", start_date, end_date)
print ("Latitude bounds", latitude_bound)
print ("IO", IO_path)
print ("-------------------------")
for dt in dates:
    d = dt.strftime("%Y-%m-%d")
    for h in hours:
        fname = satellite + '_'+str(d)+'_'+str(h)+'H_'+str(latitude_bound)+'L_'+str(tolerance)+'T.pkl'
        print(fname)
        df = pipeline(d,h,satellite,latitude_bound,ERA_fields,tolerance,method)
        df.to_pickle(IO_path+fname)
