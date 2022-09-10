#Internal
from telnetlib import VT3270REGIME
from xml.etree.ElementInclude import include
from utils.config import Config 

#external
import glob
import pandas as pd 
import numpy as np
import xarray as xr
import sys
class PrepareMLData():

    """
    Class which takes the joined monthly ERA-MODIS data and puts it in a 'nice form'
    ready for training a model.

    'Greedy' method produce a single file for each train/validate/test.

    'Sensible' method still needs to be implemented here from scripts.

    Also calculates the "delta features" i.e. V20 - V15 for the time constant features.

    Data is also normalised w.r.t training data.

    """

    def __init__(self,cfg):         
        self.config = Config.from_json(cfg)                         # Configuration file
        
        #Train/Validate/Test
        self.training_years     = self.config.data.training_years
        self.validation_years    = self.config.data.validation_years
        self.test_years          = self.config.data.test_years

        #ERA/MODIS files to take in
        self.path_to_input_data  = self.config.data.path_to_joined_ERA_MODIS_files
        self.IO_prefix           = self.config.data.IO_prefix
        
        #Extra bonus data that needs to be joined on
        #self.bonus_data          = self.config.data.bonus_data

        #Categorise different columns
        self.xt                     = self.config.data.list_of_meta_features #Time/space
        self.time_variable_features = self.config.data.list_of_time_variable_features
        self.V15_features           = self.config.data.list_of_V15_features
        self.V20_features           = self.config.data.list_of_V20_features
        #self.bonus_features        = self.config.data.list_of_bonus_features
        self.target                 = self.config.data.target_variable
        self.columns_to_load        = self.time_variable_features + self.V15_features + self.V20_features + self.target

        #Declare global emptys
        self.normalisation_mean = None 
        self.normalisation_std = None 
        self.drop_cols = None 

        #Checks
        assert len(self.V15_features) == len(self.V20_features)
        assert len(self.target) == 1 



    def _calculate_delta_fields(self,df):
        
        """Function to determine all the delta fields: V20 - V15 
           V20 fields is reassigned to a delta field."""
        
        for i in range(len(self.V15_features)):
            v15 = self.V15_features[i]
            v20 = self.V20_features[i]
            assert v20.split('_')[0] == v15.split('_')[0]

            df[v20] = df[v20] - df[v15] # Reassign the v20 fields to all be delta fields 
                    
        return df      





    def _process_year(self,years_to_process):

        """
        Function to process a list of directory of monthly files and write them to a single file.
        Also calculates normalisation over the entire training set and determines delta corrections
        Writes a single file to directory/
        """

        pop_cols = self.target+self.xt # These columns will not be popped of and won't be normalized, but will be saved to file for the test set
        unneeded_columns = ['latitude_MODIS','longitude_MODIS', 'heightAboveGround', 'H_distance_km'] # We have no need of these cols. They will be loaded but immediately dropped

        

        monthly_files = []
        for i in years_to_process:
            files = glob.glob(self.path_to_input_data+self.IO_prefix+f'*_{i}_*.parquet')
            monthly_files.append(files)
    
        monthly_files = sorted([item for sublist in monthly_files for item in sublist]) 

    
        dfs_features = [] #array to hold dfs which have features
        dfs_targets = []
        for m in monthly_files:
            print ('Loading file f:',m)
            df = pd.read_parquet(m).reset_index()
            df=df.drop(unneeded_columns,axis=1)

            #Pass monthly clake as a v20 correction
            df['clake_monthly_value'] = df['clake_monthly_value'] - df['cl_v20'] 
            df['clake_monthly_value'] =  df['clake_monthly_value'].clip(lower=0)
            assert (df['clake_monthly_value'] >= 0).all() # the monthly cl corrections should always be positive

            #Calculate delta fields
            df = self._calculate_delta_fields(df)

            #Create a target df which has just the pop cols
            df_target = pd.concat([df.pop(x) for x in pop_cols], axis=1)
            df_target['skt_unnormalised'] = df['skt']
            
            #Append 
            dfs_features.append(df)
            dfs_targets.append(df_target)
       
        
        print('All dfs loaded and processed. Now concatenate together.')
        df_features = pd.concat(dfs_features)
        df_targets = pd.concat(dfs_targets)

        if (self.normalisation_mean is None) & (self.normalisation_std is None): # All files are normalized according to the first year. Up to now that has been solely 2016

            # Check for useless columns and drop them
            print ('Checking for useless cols')
            columns_with_zero_variance = df_features.nunique()[df_features.nunique() == 1].index.values
            print (f'The following features have zero variance in year {years_to_process} and will be dropped')
            print (columns_with_zero_variance)
            self.drop_cols = columns_with_zero_variance




            print ('Calculating normalisation parameters for years:', years_to_process)
            #If we dont have any normalisation parameters already 
            self.normalisation_mean =  df_features.mean()
            self.normalisation_std =  df_features.std()


        #Normalise training features using the pre calculated terms
        df_features = (df_features-self.normalisation_mean)/self.normalisation_std

        #Get rid of columns with zero variance
        df_features = df_features.drop(self.drop_cols, axis=1)
        
        #If not the test set, only carry the MODIS_LST
        #if not include_xt:
        #    df_targets = df_targets[self.target] 

        #Concat all together
        df_out = pd.concat([df_features,df_targets],axis=1)

      
        # Save it to disk
        fout = self.path_to_input_data + '-'.join(years_to_process) + '_MLS.parquet' # Possible to save multiple years to one file, might be more sensible to just process year-by-year
        print ('Saving to:',fout)
        df_out.to_parquet(fout,compression=None)



    def greedy_preprocessing(self):

        """
        Process the ERA-MODIS data in a greedy manner, ignoring any potential future restrictions on memory
        """

        # self.path_to_input_data  = self.config.data.path_to_joined_ERA_MODIS_files
        # self.IO_prefix           = self.config.data.IO_prefix

        # all_monthly_files = sorted(glob.glob(self.path_to_input_data+self.IO_prefix+'*'))

        # years = np.array_split(all_monthly_files, len(all_monthly_files)/12)

        # for months in years:
        #     self._process_year(months)
        
        


        # print(all_monthly_files)

        #print ('Prepare training data')
        self._process_year(self.training_years)  
        self._process_year(self.validation_years) 
        self._process_year(self.test_years) 
        #print ('Prepare validation data')
        #self._process_year(self.validation_years,include_xt=True) 
    
        #print ('Prepare test data')
        #self._process_year(self.test_years,include_xt=True) 


    def sensible_preprocessing(self):

        """
        Process the ERA-MODIS data in a sensible memory-lite way, using TFRecords
        """