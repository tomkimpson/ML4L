#Internal
from telnetlib import VT3270REGIME
from xml.etree.ElementInclude import include
from utils.config import Config 

#external
import glob
import pandas as pd 

class PrepareMLData():

    """
    Class which takes the joined monthly ERA-MODIS data and puts it in a nice form
    ready for training a model.Assumes monthly data is already separated into train/validate/test folders.

    Also calculates the "delta features" i.e. V20 - V15 for the time constant features
    
    'Greedy' method produce a single file for each train/validate/test.
    """

    def __init__(self,cfg):         
        self.config = Config.from_json(cfg)                         # Configuration file
        
        #self.training_dir = self.config.data.path_to_training_data
        #self.validation_dir = self.config.data.path_to_validation_data
        #self.test_dir = self.config.data.path_to_test_data
        self.training_years = self.config.data.training_years
        self.validation_years = self.config.data.validation_years
        self.test_years = self.config.data.test_years
        self.path_to_input_data = self.config.data.path_to_joined_ERA_MODIS_files

        self.xt = self.config.data.list_of_meta_features
        self.time_variable_features = self.config.data.list_of_time_variable_features
        self.V15_features = self.config.data.list_of_V15_features
        self.V20_features = self.config.data.list_of_V20_features
        self.bonus_features = self.config.data.list_of_bonus_features
        self.target = self.config.data.target_variable

        self.columns_to_load = self.time_variable_features + self.V15_features + self.V20_features + self.bonus_features + self.target

        self.normalisation_mean = None 
        self.normalisation_std = None 


        assert len(self.V15_features) == len(self.V20_features)
        assert len(self.target) == 1 



    def _calculate_delta_fields(self,df):
        
        """Function to determine all the delta fields: V20 - V15 
           V20 fields is reassigned to a delta field."""
        
        for i in range(len(self.V15_features)):
            v15 = self.V15_features[i]
            v20 = self.V20_features[i]
            assert v20.split('_')[0] == v15.split('_')[0]

            df[v20] = df[v20] - df[v15] #Reassign the v20 fields to all be delta fields 
                    
        return df      





    def _process_year(self,years_to_process,include_xt):

        """
        Function to process a directory of monthly files and write them to a single file.
        Also calculates normalisation over the entire training set and determines delta corrections
        Writes a single file to directory/
        """

        if include_xt: #also load and carry time and position
            loaded_cols = self.columns_to_load+self.xt
            pop_cols = self.target+self.xt

        else:
            loaded_cols = self.columns_to_load
            pop_cols = self.target

    

        monthly_files = []
        for i in years_to_process:
            files = glob.glob(self.path_to_input_data+f'Haversine_MODIS_ERA_{i}_*.parquet')
            monthly_files.append(files)
    
        monthly_files = sorted([item for sublist in monthly_files for item in sublist]) 

        dfs_features = [] #array to hold dfs which have features
        dfs_targets = []
        for m in monthly_files:
            print ('Loading file f:',m)
            
            df = pd.read_parquet(m,columns=loaded_cols)
            df_target = pd.concat([df.pop(x) for x in pop_cols], axis=1)

            #Pass monthly clake as a v20 correction
            df['clake_monthly_value'] = df['clake_monthly_value'] - df['cl_v20']

            #Calculate delta fields
            df = self._calculate_delta_fields(df)
             
            #Append 
            dfs_features.append(df)
            dfs_targets.append(df_target)
       
        print('All dfs loaded and processed. Now concatenate')
        df_features = pd.concat(dfs_features)
        df_targets = pd.concat(dfs_targets)

        if (self.normalisation_mean is None) & (self.normalisation_std is None): 

            print ('Calculating normalisation parameters for years:', years_to_process)
            #If we dont have any normalisation parameters already 
            self.normalisation_mean =  df_features.mean()
            self.normalisation_std =  df_features.std()

        #Normalise it using the pre calculated terms
        df_features = (df_features-self.normalisation_mean)/self.normalisation_std

        # Concat with the targets variable which is unnormalised
        df_out = pd.concat([df_features,df_targets],axis=1)
        assert len(loaded_cols) == len(df_out.columns) #check no cols lost in the process
        
        # Save it to disk
        fout = self.path_to_input_data + '-'.join(years_to_process) + '_ML.parquet' # Possible to save multiple yeats to one file, might be more sensible to just process year-by-year
        print ('Saving to:',fout)
        df_out.to_parquet(fout,compression=None)



    def greedy_preprocessing(self):

        """
        Process the ERA-MODIS data in a greedy manner, ignoring any potential future restrictions on memory
        """

        print ('Prepare training data')
        self._process_year(self.training_years,include_xt=False)  

        
        print ('Prepare validation data')
        self._process_year(self.validation_years,include_xt=False) 
    
        # print ('Prepare test data')
        # self._process_directory(self.test_dir,include_xt=True) 


    def sensible_preprocessing(self):

        """
        Process the ERA-MODIS data in a sensible memory-lite way, using TFRecords
        """