import xarray as xr
import rioxarray
import pandas as pd
import numpy as np

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

import warnings
warnings.simplefilter("ignore")



def get_satellite_slice(date : str,
                        utc_hour : int,
                        satellite : str,
                        latitude_bound = None #Recommend only using |lat| < 70 degrees
                        ):
    
    """Function to load hourly slice of MODIS data from Mat Chantry.
       Some naming changes from original, logic msotly unchanged.
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
        #sys.exit('Exiting - there is no appropriate data')
      
    #Apply mask to data array
    sat_xr_filtered= sat_xr.where(mask,drop=True).load() 
     
    #Close
    sat_xr.close()
    sat_xr = None
    
    
   
    return sat_xr_filtered[0,::-1,:]




def get_era_data(date : str,
                 utc_hour : int,
                 field : str,
                 bounds : dict,
                 source: str):

    """Function to load hourly slice of ERA data from Mat Chantry.
       Additional filtering by max/min longitude, read from matching MODIS file
       Some naming changes from original. """
        
    #Load the data
    month = '_'.join(date.split('-')[:-1])
    
    if source == 'ERA_skin':
        name = '_skin_'
    if source == 'ERA_skin2':
        name = '_skin2_'
    if source == 'ERA_sfc':
        name = '_'
        
        
    fname = f'{era_folder}/{source}/sfc{name}unstructured_{month}.grib'
 #   print(fname)
    ds_era = xr.open_dataset(fname,engine='cfgrib',backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}, 'indexpath': ''}) #supress index files

    
    #Grab correct field
    if field is not None:
        da = ds_era[field]
    else:
        da = ds_era
        
        

    #Grab the correct time
    time_str = f"{date} {utc_hour:02}:00:00" 
    da = da.sel(time=time_str)
    
    
    #Relabel longitude coordinate to be consistent with MODIS
    da = da.assign_coords({"longitude": (((da.longitude + 180) % 360) - 180)})
    
    
    # Also filter by latitude/longtiude
    longitude_filter = (da.longitude > bounds['longitude_min']) & (da.longitude < bounds['longitude_max'])
    latitude_filter =  (da.latitude > bounds['latitude_min']) & (da.latitude < bounds['latitude_max'])
    
  
    da_filtered = da.where(longitude_filter & latitude_filter,drop=True)


      
    #Close file, attempt to not have memory leaks
    ds_era.close()
    ds_era = None
    
    return da_filtered.load()


    
    
def find_closest_match_sklearn(MODIS_data, ERA_data,tolerance):
    
     
    MODIS1 = MODIS_data.to_dataframe(name='MODIS_LST').reset_index()
    ERA1_selected = ERA_data.to_dataframe().reset_index()

    
    NN = NearestNeighbors(n_neighbors=1, metric='haversine') #algorithm = balltree, kdtree or brutie force


    NN.fit(np.deg2rad(MODIS1[['latitude', 'longitude']].values))


    query_lats = ERA1_selected['latitude'].astype(np.float64)
    query_lons = ERA1_selected['longitude'].astype(np.float64)
    X = np.deg2rad(np.c_[query_lats, query_lons])
    distances, indices = NN.kneighbors(X, return_distance=True)


    r_km = 6371 # multiplier to convert to km (from unit distance)
    distances = distances*r_km


    
    df_combined = ERA1_selected.reset_index().join(MODIS1.iloc[indices.flatten()].reset_index(), lsuffix='_ERA',rsuffix='_MODIS')
    df_combined['distance'] = distances
    df_combined['MODIS_idx'] = indices

    #Filter and surface selected columns
    df_combined_matches = df_combined.query('distance < %.9f' % tolerance)

    
    return df_combined_matches.dropna().reset_index()    
    
    


def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)
    


    
    
    
    
def pipeline(date,utc_hour,satellite,latitude_bound,ERA_fields,tolerance):

    # MODIS data
    MODIS = get_satellite_slice(date,utc_hour,satellite,latitude_bound)
  
    if not isinstance(MODIS,xr.DataArray):
        return pd.DataFrame()
  
    
    #ERA data
    delta = 0.5
    bounds = {"latitude_min" :MODIS.latitude.data.min()-delta,
          "latitude_max" :MODIS.latitude.data.max()+delta,
          "longitude_min":MODIS.longitude.data.min()-delta,
          "longitude_max":MODIS.longitude.data.max()+delta
          }

    ERA_sfc = get_era_data(date, utc_hour, field=ERA_fields,bounds=bounds,source='ERA_sfc')
    ERA_skin = get_era_data(date, utc_hour, field=ERA_fields,bounds=bounds,source='ERA_skin')
    ERA = xr.merge([ERA_sfc, ERA_skin])
    
    #Combine
    df_sk = find_closest_match_sklearn(MODIS,ERA,tolerance)
    
    #Deallocate everything
    ERA_sfc.close()
    ERA_skin.close()
    ERA_sfc = None
    ERA_skin = None
    
    return df_sk




#Parameters
start_date = date(2018, 1, 2)
end_date   = date(2020, 12, 30)
dates = daterange(start_date,end_date)

hours = np.arange(0,24)


satellite='aquaDay'
latitude_bound=70
ERA_fields = None 
tolerance = 10 #km

#Path for where to ouput saved files
IO_path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/outputs_multi/'

for dt in dates:
    d = dt.strftime("%Y-%m-%d")
    for h in hours:
        fname = satellite + '_'+str(d)+'_'+str(h)+'H_'+str(latitude_bound)+'L_'+str(tolerance)+'T.pkl'
        print(fname)
        df = pipeline(d,h,satellite,latitude_bound,ERA_fields,tolerance)
        df.to_pickle(IO_path+fname)
