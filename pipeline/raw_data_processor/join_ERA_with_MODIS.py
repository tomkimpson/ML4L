#Internal
from utils.config import Config 
from utils.utils import get_list_of_files


#External
import xarray as xr
import glob
import numpy as np
import pandas as pd
from contextlib import suppress
#import faiss
from sklearn.neighbors import NearestNeighbors
from cuml.neighbors import NearestNeighbors as cumlNearestNeighbours
import cudf

import sys

class JoinERAWithMODIS():

    """
    Class which takes all the MODIS data and all the ERA data and matches in time and space.
    """

    def __init__(self,cfg):         
        self.config = Config.from_json(cfg)                         # Configuration file
        

        self.ERA_constants_dict = {}
        self.V15_output_path = self.config.data.path_to_processed_V15_climate_fields
        self.V20_output_path = self.config.data.path_to_processed_V20_climate_fields
        self.monthly_clake_files_path = self.config.data.path_to_monthly_clake_files
        self.saline_clake_files_path = self.config.data.path_to_saline_clake_files


        self.monthly_clake_ds = xr.Dataset() #Empty ds to hold monthly clake values
        self.saline_ds = None # Empty declaration ready to hold saline lake fields
        self.ERA_files = sorted(get_list_of_files(self.config.data.path_to_processed_variable_fields,self.config.data.min_year_to_join,self.config.data.max_year_to_join))
            
            
    


        self.min_hours = {"aquaDay":    self.config.data.aquaDay_min_hour,
                          "terraDay":   self.config.data.terraDay_min_hour,
                          "aquaNight":  self.config.data.aquaNight_min_hour,
                          "terraNight": self.config.data.terraNight_min_hour }

        self.max_hours = {"aquaDay":    self.config.data.aquaDay_max_hour,
                          "terraDay":   self.config.data.terraDay_max_hour,
                          "aquaNight":  self.config.data.aquaNight_max_hour,
                          "terraNight": self.config.data.terraNight_max_hour}


        self.local_times = {"aquaDay":   self.config.data.aquaDay_local_solar_time,
                            "terraDay":   self.config.data.terraDay_local_solar_time,
                            "aquaNight":  self.config.data.aquaNight_local_solar_time,
                            "terraNight": self.config.data.terraNight_local_solar_time}

        self.satellite = self.config.data.satellite
        self.satellite_folder = self.config.data.path_to_MODIS_data
        self.previous_datestring = None

        self.latitude_bound = self.config.data.latitude_bound
        self.IO_path = self.config.data.path_to_joined_ERA_MODIS_files


        self.joining_metric =  self.config.data.joining_metric




    def _load_constant_ERA_data(self,f,v):

        """
        Load a version of the constant V15/V20 ERA data and append it to a global dictionary
        """

        ds = xr.open_dataset(f) #NetCDF file of features which are constant for each gridpoint
        
        name_dict={x:x+f'_{v}' for x in list(ds.keys())}
        ds = ds.rename(name_dict)
    
        self.ERA_constants_dict[v] = ds
        ds.close()
        

    def _load_monthly_clake_data(self):

        """
        Load a version of the constant V15/V20 ERA data and append it to a global dictionary
        """

        monthly_clake_files = sorted(glob.glob(self.monthly_clake_files_path+'clake*'))
        month_counter = 1
        
        for m in monthly_clake_files:
            print(m)
            ds_clake= xr.open_dataset(m,engine='cfgrib',backend_kwargs={'indexpath': ''}) 
            
            #Rename the parameter so everything is not cldiff
            ds_clake = ds_clake.cl.rename(f'clake_monthly_value') #This is now a dataarray
            
            #Fix the time to be an integer
            ds_clake['time'] = month_counter #i.e. what month it it? An integer between 1 and 12
            
            
            #Append this to dataset
            self.monthly_clake_ds[f"month_{month_counter}"] = ds_clake 
            month_counter += 1
           # monthly_clake_ds is a dataset where each variable is month_1, month_2 etc. representing a global field for that time
           # Later on we will select just the correspondig month
    
    def _load_saline_lake_data(self):

        """
        Load a the clake saline data
        """
        self.saline_ds = xr.open_dataset(self.saline_clake_files_path,engine='cfgrib',backend_kwargs={'indexpath': ''})
        self.saline_ds = self.saline_ds.cl.rename(f'cl_saline') #This is now a data array
        self.saline_ds = self.saline_ds.to_dataset() #This is now a dataset
    
    def _select_correct_MODIS_file(self,t):

        """We have to be careful with the dateline. This function
            figures out which MODIS file to load."""

        #Get the hour
        utc_hour = t.hour
        
        
        #Due to crossing of the datetime, some times will be saved different date
        if utc_hour < self.min_hours[self.satellite]:
            file_date = t  - np.timedelta64(1,'D')
        elif utc_hour > self.max_hours[self.satellite]:
            file_date = t  + np.timedelta64(1,'D')
        else:
            file_date = t
            
        #Create a string which will be used to open file
        y = pd.to_datetime(file_date).year
        m = pd.to_datetime(file_date).month
        d = pd.to_datetime(file_date).day
        date_string = f'{y}-{m:02}-{d:02}'
        
        return date_string
    
    def _load_MODIS_file(self,date_string):
        
        """
        Load a day of MODIS data, apply some filters and corrections
        """
        
        #Open that file
        MODIS_data = xr.open_dataarray(f'{self.satellite_folder}/{self.satellite}_errorGTE03K_04km_{date_string}.tif',engine="rasterio")

        #Make some edits to file
        MODIS_data = MODIS_data.rename({'x':'longitude','y':'latitude'})

        #Filter by latitude bound
        space_filter = np.expand_dims(np.abs(MODIS_data.latitude) < self.latitude_bound,axis=(0,-1))
        mask = np.logical_and(np.isfinite(MODIS_data),space_filter) #make it a 2d mask
        MODIS_data = MODIS_data.where(mask,drop=True)

        #Convert local satellite time to UTC and round to nearest hour
        time_delta = pd.to_timedelta(MODIS_data.longitude.data/15,unit='H') 
        time_UTC = (pd.to_datetime([date_string + " " + self.local_times[self.satellite]]*time_delta.shape[0]) - time_delta).round('H')

        return MODIS_data,time_UTC


    def _get_ERA_hour(self,ERA_month,t,clake_month,bounds):
        
        """
        Extract an hour of ERA data
        """
        
        #Filter month of ERA data to an hour
        time_filter = (ERA_month.time == t)
        ERA_hour = ERA_month.where(time_filter,drop=True)
        
        #Grab the constant fields and make a local copy
        v15 = self.ERA_constants_dict['v15']
        v20 = self.ERA_constants_dict['v20']
        saline = self.saline_ds

        #Join on the constant data V15 and v20, the monthly clake files, and the saline data, first setting the time coordinate to allow for merge
        v15 = v15.assign_coords({"time": (((ERA_hour.time)))}) 
        v20 = v20.assign_coords({"time": (((ERA_hour.time)))}) 
        clake_month = clake_month.assign_coords({"time": (((ERA_hour.time)))})
        saline = saline.assign_coords({"time": (((ERA_hour.time)))}) 

  
        ERA_hour = xr.merge([ERA_hour,v15,v20,clake_month,saline]).load() #Explicitly load 
        
        
        #And covert longitude to long1
        ERA_hour = ERA_hour.assign_coords({"longitude": (((ERA_hour.longitude + 180) % 360) - 180)})

        # Also filter by latitude/longtiude
        longitude_filter = (ERA_hour.longitude > bounds['longitude_min']) & (ERA_hour.longitude < bounds['longitude_max'])
        latitude_filter =  (ERA_hour.latitude > bounds['latitude_min']) & (ERA_hour.latitude < bounds['latitude_max'])
        return ERA_hour.where(longitude_filter & latitude_filter,drop=True)
    

    def _haver(self,lat1_deg,lon1_deg,lat2_deg,lon2_deg):
        
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
    

    def _faiss_knn(self,database,query):
        
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
        df['H_distance'] = self._haver(df['latitude_MODIS'],df['longitude_MODIS'],df['latitude_ERA'],df['longitude_ERA']) #Haversine distance
        
        #Filter out any large distances
        tolerance = 50 #km
        df_filtered = df.query('H_distance < %.9f' % tolerance)

        #Group it. Each ERA point has a bunch of MODIS points. Group and average
        df_grouped = df_filtered.groupby(['latitude_ERA','longitude_ERA'],as_index=False).mean()

        
        return df_grouped


    
    def _find_closest_match_rapids(self,database,query):

        print ('Finding closest match using RAPIDs GPU method')
               
        #Construct NN     
        NN = cumlNearestNeighbours(n_neighbors=1,algorithm='brute',metric='haversine')
        X = np.deg2rad(database[['latitude', 'longitude']].values).astype('float32')
        NN.fit(X)
        print ('Input shape:', X.shape)
        #-------------------------------

        #query_lats = query['latitude'].astype(np.float32)
        #query_lons = query['longitude'].astype(np.float32)
        Xq = np.deg2rad(query[['latitude', 'longitude']].values).astype('float32')
        print ('Test shape:', Xq.shape)
        #Xq = np.deg2rad(np.c_[query_lats, query_lons])
        



        X_cudf = cudf.DataFrame(Xq) #Make it a cudf


       #--------------------------------
   
        distances, indices = NN.kneighbors(X_cudf, return_distance=True)
        
        #print ('Matches found')
        distances = distances.to_array()
        indices = indices.to_array()
        
        r_km = 6371 # multiplier to convert to km (from unit distance)
        distances = distances*r_km

        df = query.reset_index().join(database.iloc[indices].reset_index(), lsuffix='_MODIS',rsuffix='_ERA')
        df['H_distance'] = distances
        
        print ('This is the joined df before any filtering')
        print (df.columns)
        print(df)

        #Filter out any large distances
        tolerance = 50 #km #MOVE THIS TO CONFIG
        df_filtered = df.query('H_distance < %.9f' % tolerance)

        print ('This is the joined df after filtering')
        print(df_filtered)

        #Group it. Each ERA point has a bunch of MODIS points. Group and average
        df_grouped = df_filtered.groupby(['latitude_ERA','longitude_ERA']).mean()
        df_grouped['number_of_modis_observations'] = df_filtered.value_counts(subset=['latitude_ERA','longitude_ERA']) # Must be a way to combine this with the above line. Can used grouped agg, but then need to specify operation for each column?
        df_grouped = df_grouped.reset_index()    


        print ('This is the grouped output df')
        #Drop any columns we dont care about
        df_grouped = df_grouped.drop(['index_MODIS', 'band','spatial_ref','index_ERA','values','number','surface','depthBelowLandLayer'], axis=1) #get rid of all these columns that we dont need
        print(df_grouped)

        sys.exit()

        return df_grouped
    
    
    def _find_closest_match_sklearn(self,database,query):
               
        #Construct NN     
       # NN = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', leaf_size=60,metric='haversine') #algorithm = balltree, kdtree or brutie force

        NN = cumlNearestNeighbours(n_neighbors=1,metric='haversine')
        
        NN.fit(np.deg2rad(database[['latitude', 'longitude']].values))

        print ('FIT COMPLETED OK')

        query_lats = query['latitude'].astype(np.float64)
        query_lons = query['longitude'].astype(np.float64)


        X = np.deg2rad(np.c_[query_lats, query_lons])
        print ('NOW QUERY')
        print ('X =', X)
        distances, indices = NN.kneighbors(X, return_distance=True)


        r_km = 6371 # multiplier to convert to km (from unit distance)
        distances = distances*r_km

        df = query.reset_index().join(database.iloc[indices.flatten()].reset_index(), lsuffix='_MODIS',rsuffix='_ERA')
        df['H_distance'] = distances
        
        #Filter out any large distances
        tolerance = 50 #km
        df_filtered = df.query('H_distance < %.9f' % tolerance)

        #Group it. Each ERA point has a bunch of MODIS points. Group and average
        df_grouped = df_filtered.groupby(['latitude_ERA','longitude_ERA']).mean()
        df_grouped['counts'] = df_filtered.value_counts(subset=['latitude_ERA','longitude_ERA']) # Must be a way to combine this with the above line. Can used grouped agg, but then need to specify operation for each column?

        return df_grouped


    def join(self):

        #Load the constant ERA fields and append to dictionary self.ERA_constants_dict
        self._load_constant_ERA_data(self.V15_output_path,"v15")
        self._load_constant_ERA_data(self.V20_output_path,"v20")

        #Load the monthly clake files
        self._load_monthly_clake_data()

        #Load the saline lake
        self._load_saline_lake_data()
        print('Iterating over the following months:',self.ERA_files[0],self.ERA_files[11])
        for f in [self.ERA_files[0],self.ERA_files[11] ]: #Iterate over all months
            #Load a month of ERA data
            print ('Loading ERA month:', f)
            ERA_month = xr.open_dataset(f,engine='cfgrib',backend_kwargs={'indexpath': ''})

            #Get all times in that month of data. These are on an hourly grain
            timestamps = pd.to_datetime(ERA_month.time) 

            dfs = []
            for t in timestamps: #iterate over every time (hour)

                print(t)

                #First grab the clake bonus data for that month
                #Note that we do this every timestamp, rather just doing it once per ERA month since ERA month sometimes
                #contains values over two months e.g. all of February and the first day of March.
                #There may be a more efficeint work around but these 4 lines are very inexpensive so can stay here for now. 
                clake_month = self.monthly_clake_ds[f"month_{t.month}"]   
                clake_month = clake_month.to_dataset()                 
                clake_month['clake_monthly_value'] = clake_month[f"month_{t.month}"] # Rename data variable by declaring a new entry... 
                clake_month = clake_month.drop([f"month_{t.month}"])                 # ...and dropping the old one



                date_string = self._select_correct_MODIS_file(t) #For this datetime, which MODIS file should be opened? 
                if date_string == '2015-12-31': continue # skip since we don't have data this far back


                if date_string != self.previous_datestring:
                    # We need to open a new file. 
                    with suppress(NameError):MODIS_data.close() #First close the old one explicitly. Exception handles case where MODIS_data not yet defined
                    MODIS_data,time_UTC = self._load_MODIS_file(date_string)
                    self.previous_datestring = date_string


                # Filter to only select the hour of data we want
                time_filter = np.expand_dims(time_UTC == t,axis=(0,1))
                mask = np.logical_and(np.isfinite(MODIS_data),time_filter)
                MODIS_hour = MODIS_data.where(mask,drop=True).load() 
                MODIS_df = MODIS_hour.to_dataframe(name='MODIS_LST').reset_index().dropna() #Make everything a pandas df to pass into faiss_knn. Unnecessary step?


                if MODIS_df.empty:
                    print('MODIS dataframe is empty for t = ', t)
                    print('Skipping to next timestep')
                    continue


                #Spatial bounds
                #Get the limits of the MODIS box. We will use this to filter the ERA data for faster matching
                #i.e. when looking for matches, dont need to look over the whole Earth, just a strip
                delta = 1.0 # Enveloping box
                bounds = {"latitude_min" : MODIS_df.latitude.min()-delta,
                        "latitude_max" :   MODIS_df.latitude.max()+delta,
                        "longitude_min":   MODIS_df.longitude.min()-delta,
                        "longitude_max":   MODIS_df.longitude.max()+delta
                }

                # Get an hour of ERA data
                ERA_hour = self._get_ERA_hour(ERA_month,t,clake_month,bounds) # Get an hour of all ERA data
                ERA_df = ERA_hour.to_dataframe().reset_index()                # Make it a df

                #Find matches in space
                if self.joining_metric == 'haversine':
                    #df_matched = self._find_closest_match_sklearn(ERA_df,MODIS_df)
                    df_matched = self._find_closest_match_rapids(ERA_df,MODIS_df)
                elif self.joining_metric == 'L2':
                    df_matched = self._faiss_knn(ERA_df,MODIS_df) 
                else:
                    sys.exit(f'Joining metric {self.joining_metric} is not a valid option')



                df_matched['time'] = t            
                df_matched = df_matched.drop(['index_MODIS', 'band','spatial_ref','index_ERA','values','number','surface','depthBelowLandLayer'], axis=1) #get rid of all these columns that we dont need
                dfs.append(df_matched)

               
                # Explicitly deallocate
                ERA_hour.close()
                MODIS_hour.close()
        
            # At the end of every month, do some IO
            df = pd.concat(dfs)
            year_month = f.split('/')[-1].split('.')[0]
            fname = f'ExampleHaversine_MODIS_{year_month}.parquet'
            print ("Writing to disk:", self.IO_path+fname)
            df.to_parquet(self.IO_path+fname,compression=None)

            # Deallocate
            ERA_month.close()