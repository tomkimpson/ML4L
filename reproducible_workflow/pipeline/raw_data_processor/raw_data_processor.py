
#Internal
from utils.config import Config 

#External
import xarray as xr
import glob
import os
import shutil
import numpy as np
import tempfile
import pandas as pd
from contextlib import suppress
import faiss


class ProcessERAData():
    """
    Class to process the raw ERA data which is mixed in different files into a cleaner form.
    Two functions which can be called:
    
    * process_time_constant_data() : Creates a two NetCDF files - one for V15 and one for V20 - for the constant-in-time climate fields
    * process_time_variable_data() : Creates N monthly files for the variable-in-time fields
    
    """

    def __init__(self,cfg):         
        self.config = Config.from_json(cfg)                         # Configuration file
        
        self.constant_features = self.config.data.ERA_skin_constant_features
        self.variable_features = self.config.data.ERA_skin_variable_features
        
        self.first_year = self.config.data.min_year
        self.last_year = self.config.data.max_year

        
        #Raw
        self.ERA_sfc_path = self.config.data.path_to_raw_ERA_sfc 
        self.ERA_skin_path = self.config.data.path_to_raw_ERA_skin 
        self.ERA_skt_path = self.config.data.path_to_raw_ERA_skt 


        self.V15_path = self.config.data.path_to_raw_V15_climate_fields
        self.V20_path = self.config.data.path_to_raw_V20_climate_fields

        #Processed
        self.V15_output_path = self.config.data.path_to_processed_V15_climate_fields
        self.V20_output_path = self.config.data.path_to_processed_V20_climate_fields
        
        self.variable_output_path = self.config.data.path_to_processed_variable_fields
        
        
        #tmp
        self.tmpdir = self.config.data.tmpdir
        
    def _create_tmpdir(self):
        """Create a temporary directory to write to.
            Anything here is overwrite-able"""
        if os.path.exists(self.tmpdir): #Delete any existing dir
            shutil.rmtree(self.tmpdir)
        os.mkdir(self.tmpdir)
      
   
    def _get_list_of_files(self,directory):
        """
        Get a flattened list of all grib files within a directory within a certain time range
        The time range is read from the .grib file name.
        """
        globs_exprs = [directory+f'*_{i}_*.grib'.format(i) for i in np.arange(self.first_year, self.last_year+1)]
        list_of_files = [glob.glob(g) for g in globs_exprs]
        return sorted([item for sublist in list_of_files for item in sublist])
        
        
                
    def process_time_constant_data(self):
        
        """
        Process the time constant data.
        This involved extracting the time constant features from ERA_skin and then merging them with the V* fields.
        Outputs two NetCDF files, one for V15 and V20.
        Note that both V15 and V20 have the **same** extracted self.constant_features from ERA_skin.
        """
        
        
        print("Processing the time constant ERA data.")

        ###------------------------------------------------------------###
        ###------------Extract time constant ERA skin features---------###
        ###------------------------------------------------------------###
        
        #First extract those features from ERA skin which ARE constant but are not in the climateV15/V20 files
        
        ERA_skin = self.ERA_skin_path + 'sfc_skin_unstructured_2018_01.grib' # Select just a single ERA skin file. Does not matter which. 
        ds = xr.open_dataset(ERA_skin,engine='cfgrib',
                             filter_by_keys={'typeOfLevel': 'surface'},
                             backend_kwargs={'indexpath': ''})               # Open that file. NOTE: assumes constant fields are 'typeOfLevel': 'surface'. This is true for now, but may not always be so.
        
        selected_data = ds[self.constant_features].isel(time=[0])            # Get a snapshot of those features at a single time
        
  
        ###---------------------------------------------------###
        ###----------Deal with V15/V20 surface fields---------###
        ###---------------------------------------------------###

        #Method here, for a particular version, is to:
        #.   Get all the climate files
        #.   Split by parameter to produce a bunch of files
        #.   Load each new file, append it to array
        #.   Append in the selected_data dataset from above
        #.   Merge it all together

    
        #Dictionary mapping input paths to single output file
        climateV = {self.V15_path:self.V15_output_path,
                    self.V20_path:self.V20_output_path,
                    }
        self._create_tmpdir() #Create a tmp directory
        for v in climateV: #for each version of the climate fields
            input_path = v
            output_path = climateV[v]
            
            version_files = set(glob.glob(input_path+'*'))# Get all the grib files in that directory
            splitfile = self.tmpdir+'/splitfile_[shortName].grib'
    
            for f in version_files: #Split each file into sub-files by feature
                print(f)
                query_split = f'grib_copy {f} "{splitfile}"' 
                os.system(query_split)
        

            ds_all = []                               # Create an empty array
            ds_all.append(selected_data)              # Add the data slice you took above
    
            splitfiles = glob.glob(self.tmpdir + '*.grib') # Get a list of all the splitfiles you have just created
            for f in splitfiles: #Load each file in turn
                ds = xr.open_dataset(f,engine='cfgrib',backend_kwargs={'indexpath': ''})
                ds_all.append(ds)
                
            constant_merged_data = xr.merge(ds_all,compat='override') #need the override option to deal with keys
            constant_merged_data.to_netcdf(output_path,mode='w') #write to disk
        
 

    def process_time_variable_data(self):
        
        
        """
        Process the time variable data.
        For each month, merge ERA_sfc, ERA_skt and the time variable ERA_skin
        Outputs N monthly files
        """

        print("Processing the time variable ERA data.")

        
        #Get list of raw .grib monthly files in a specific time range
        ERA_sfc_files =  self._get_list_of_files(self.ERA_sfc_path)        
        ERA_skin_files =  self._get_list_of_files(self.ERA_skin_path)        
        ERA_skt_files =  self._get_list_of_files(self.ERA_skt_path)        

        
        
        for i in range(len(ERA_sfc_files)):
            sfc,skin,skt = ERA_sfc_files[i], ERA_skin_files[i], ERA_skt_files[i]
            y = skin.split('_')[-2] #read the year from the filename
            m = skin.split('_')[-1] #and the month.grib
            outfile  = f'{self.variable_output_path}ERA_{y}_{m}'
            
            print(outfile)
            with tempfile.NamedTemporaryFile() as tmp1, tempfile.NamedTemporaryFile() as tmp2: #Create two tmp files to write to
        
                tmpfile1,tmpfile2 = tmp1.name,tmp2.name
            
                #Extract the time variable features from ERA_skin, save to tmpfile1
                query_extract = f'grib_copy -w shortName={self.variable_features} {skin} {tmpfile1}'
                os.system(query_extract)
                
                # Deal with the istl1,2 featuresm since these are not surface levels which is annoying later on when trying to load the file
                #Set the non surface levels to same type: istl1,2.
                # tmpfile1--->tmpfile2
                query_level = f'grib_set -s level=0 -w level=7 {tmpfile1} {tmpfile2}'
                os.system(query_level)

                #Merge ERA_sfc, our processed ERA_skin tmpfile and ERA_skt into a single file
                query_merge = f'grib_copy {sfc} {tmpfile2} {skt} {outfile}'
                os.system(query_merge)



class JoinERAWithMODIS():

    """
   
    
    """
    print("Joining ERA data with MODIS data")

    def __init__(self,cfg):         
        self.config = Config.from_json(cfg)                         # Configuration file
        

        self.ERA_constants_dict = {}
        self.V15_output_path = self.config.data.path_to_processed_V15_climate_fields
        self.V20_output_path = self.config.data.path_to_processed_V20_climate_fields
        self.monthly_clake_files_path = self.config.data.path_to_monthly_clake_files


        self.monthly_clake_ds = xr.Dataset() #Empty ds
        self.ERA_files = sorted(glob.glob(self.config.data.path_to_processed_variable_fields+'*'))


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

    def _load_constant_ERA_data(self,f,v):

        ds = xr.open_dataset(f) #NetCDF file of features which are constant for each gridpoint
        
        name_dict={x:x+f'_{v}' for x in list(ds.keys())}
        ds = ds.rename(name_dict)
    
        self.ERA_constants_dict[v] = ds
        ds.close()
        

    def _load_monthly_clake_data(self):

        monthly_clake_files = sorted(glob.glob(self.monthly_clake_files_path+'clake*'))
        month_counter = 1
        
        for m in monthly_clake_files:
            print(m)
            ds_clake= xr.open_dataset(m,engine='cfgrib',backend_kwargs={'indexpath': ''}) 
            
            #Rename the parameter so everything is not cldiff
            ds_clake = ds_clake.cl.rename(f'clake_monthly_value') #This is now a dataarray
            
            #Fix the time to be an integer
            ds_clake['time'] = month_counter #i.e. what month it it? An integer between 1 and 12
            
            
            #Append this to dataframe
            self.monthly_clake_ds[f"month_{month_counter}"] = ds_clake 
            month_counter += 1
           # monthly_clake_ds is a dataset where each variable is month_1, month_2 etc. representing a global field for that time
           # Later on we will select just the correspondig month
    
    
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
        v15 = self.ERA_constant_dict['v15'], 
        v20 = self.ERA_constant_dict['v20']


        #Join on the constant data V15 and v20, and the wetlands data, first setting the time coordinate
        v15 = v15.assign_coords({"time": (((ERA_hour.time)))}) 
        v20 = v20.assign_coords({"time": (((ERA_hour.time)))}) 
        clake_month = clake_month.assign_coords({"time": (((ERA_hour.time)))})
        
        ERA_hour = xr.merge([ERA_hour,v15,v20, clake_month]).load() #Explicitly load 
        
        
        #And covert longitude to long1
        ERA_hour = ERA_hour.assign_coords({"longitude": (((ERA_hour.longitude + 180) % 360) - 180)})
        
        
            
        # Also filter by latitude/longtiude
        longitude_filter = (ERA_hour.longitude > bounds['longitude_min']) & (ERA_hour.longitude < bounds['longitude_max'])
        latitude_filter =  (ERA_hour.latitude > bounds['latitude_min']) & (ERA_hour.latitude < bounds['latitude_max'])
        return ERA_hour.where(longitude_filter & latitude_filter,drop=True)
    

    def _haver(lat1_deg,lon1_deg,lat2_deg,lon2_deg):
        
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
    

    def _faiss_knn(database,query):
        
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
        df['H_distance'] = _haver(df['latitude_MODIS'],df['longitude_MODIS'],df['latitude_ERA'],df['longitude_ERA']) #Haversine distance
        
        #Filter out any large distances
        tolerance = 50 #km
        df_filtered = df.query('H_distance < %.9f' % tolerance)

        #Group it. Each ERA point has a bunch of MODIS points. Group and average
        df_grouped = df_filtered.groupby(['latitude_ERA','longitude_ERA'],as_index=False).mean()

        
        return df_grouped



    def join(self):

        #Load the constant ERA fields and append to dictionary self.ERA_constants_dict
        self._load_constant_ERA_data(self.V15_output_path,"v15")
        self._load_constant_ERA_data(self.V20_output_path,"v20")

        #Load the monthly clake files
        self._load_monthly_clake_data()

        for f in self.ERA_files[0:1]: #Iterate over all months
            #Load a month of ERA data
            print ('Loading ERA month:', f)
            ERA_month = xr.open_dataset(f,engine='cfgrib',backend_kwargs={'indexpath': ''})
    
            #Get all times in that month of data. These are on an hourly grain
            timestamps = pd.to_datetime(ERA_month.time) 
        
    
            #Load the clake bonus data for that month. This is a clumsy method that needs cleaning up
            assert len(np.unique(timestamps.month)) == 1            # There should only be one value, an integer in range 1-12
            month = np.unique(timestamps.month)[0]                  # Select that one value        
            clake_month = self.monthly_clake_ds[f"month_{month}"]   # Get a month of data
            clake_month = clake_month.to_dataset()                  # Make it a dataset

            clake_month['clake_monthly_value'] = clake_month[f"month_{month}"] # Rename data variable by declaring a new entry... 
            clake_month = clake_month.drop([f"month_{month}"])                 # ...and dropping the old one
            

            dfs = []
            for t in timestamps: #iterate over every time (hour)

                print(t)
                date_string = self._select_correct_MODIS_file(t) #For this datetime, which MODIS file should be opened? 
                if date_string == '2017-12-31': continue #skip since we don't have this day. Would be better to replace this with self.min_year



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
                df_matched = _faiss_knn(ERA_df,MODIS_df) #Match reduced gaussian grid to MODIS
                df_matched['time'] = t            
                df_matched = df_matched.drop(['index_MODIS', 'band','spatial_ref','index_ERA','values','number','surface','depthBelowLandLayer'], axis=1) #get rid of all these columns that we dont need
                dfs.append(df_matched)