
#Internal
from locale import ERA
from utils.config import Config 
from utils.utils import get_list_of_files

#External
import xarray as xr
import glob
import os
import shutil
import numpy as np
import tempfile


import sys

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
        
        self.first_year = self.config.data.min_year_to_process
        self.last_year = self.config.data.max_year_to_process

        
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
  
        with tempfile.TemporaryDirectory() as tmpdir:
            for v in climateV: #for each version of the climate fields
                input_path = v
                output_path = climateV[v]
                version_files = set(glob.glob(input_path+'*'))# Get all the grib files in that directory
                splitfile = tmpdir+'/splitfile_[shortName].grib'
                for f in version_files: #Split each file into sub-files by feature
                    print(f)
                    query_split = f'grib_copy {f} "{splitfile}"' 
                    os.system(query_split)
            

                ds_all = []                               # Create an empty array
                ds_all.append(selected_data)              # Add the data slice you took above
        
                splitfiles = glob.glob(tmpdir + '/*.grib') # Get a list of all the splitfiles you have just created
                for f in splitfiles: #Load each file in turn
                    ds = xr.open_dataset(f,engine='cfgrib',backend_kwargs={'indexpath': ''})
                    ds_all.append(ds)
                    
                constant_merged_data = xr.merge(ds_all,compat='override') # need the override option to deal with keys
                constant_merged_data.to_netcdf(output_path,mode='w')      # write to disk
    

    def process_time_variable_data(self):
        
        
        """
        Process the time variable data.
        For each month, merge ERA_sfc, ERA_skt and the time variable ERA_skin
        Outputs N monthly files
        """

        print("Processing the time variable ERA data in the range: ",self.first_year,self.last_year)

        
        #Get list of raw .grib monthly files in a specific time range
        ERA_sfc_files  =  get_list_of_files(self.ERA_sfc_path,2018,self.last_year)        
        ERA_skin_files =  get_list_of_files(self.ERA_skin_path,2018,self.last_year)        
        ERA_skt_files  =  get_list_of_files(self.ERA_skt_path,2018,self.last_year)        
        
        #for i in range(len(ERA_sfc_files)):
        for i in range(1):
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


                print ('Loading newly created outfile')
                ds = xr.open_dataset(outfile,engine='cfgrib',backend_kwargs={'indexpath': ''})
                print(ds)


