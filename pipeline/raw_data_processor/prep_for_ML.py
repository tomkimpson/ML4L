#Internal
from telnetlib import VT3270REGIME
from utils.config import Config 

#external
import glob
import pandas as pd 

class PrepareMLData():

    """
    Class which takes the joined monthly ERA-MODIS data and puts it in a nice form
    ready for training a model.Assumes monthly data is already separated into train/validate/test folders.

    Also calculates the "delta features" i.e. V20 - V15 for the time constant features
    
    Greedy produce a single file for each train/validate/test.
    """

    def __init__(self,cfg):         
        self.config = Config.from_json(cfg)                         # Configuration file
        
        self.training_dir = self.config.data.path_to_training_data
        self.validation_dir = self.config.data.path_to_validation_data


        self.time_variable_features = self.config.data.list_of_time_variable_features
        self.V15_features = self.config.data.list_of_V15_features
        self.V20_features = self.config.data.list_of_V20_features
        self.bonus_features = self.config.data.list_of_bonus_features
        self.target = self.config.data.target_variable

        self.features = self.time_variable_features + self.V15_features + self.V20_features + self.bonus_features #+ self.target

        self.normalisation_mean = None 
        self.normalisation_std = None 


        assert len(self.V15_features) == len(self.V20_features)


    def _calculate_delta_fields(self,df):
        
        """Function to determine all the delta fields: V20 - V15 
           Later on we will drop"""
        
        for i in range(len(self.V15_features)):
            v15 = self.V15_features[i]
            v20 = self.V20_features[i]
            print(v20,v15)

            df[v20] = df[v20] - df[v15] #Reassign the v20 fields to all be delta fields 
                    
        return df      





    def _process_directory(self,directory):

        monthly_files = glob.glob(directory+'/*.parquet')

        dfs_features = [] #array to hold dfs which have features
        dfs_targets = []
        for m in monthly_files:
            print ('Loading file f:')

            df = pd.read_parquet(m,columns=self.features)
            print(df.columns)

            #Pass monthly clake as a v20 correction
            df['clake_monthly_value'] = df['clake_monthly_value'] - df['cl_v20']

            #Calculate delta fields
            df = self._calculate_delta_fields(df)

            dfs_features.append(df)

            #Also load target variable separatley
            df_target = pd.read_parquet(m,columns=self.target)
            dfs_targets.append(df_target)
       
        print('All files processed. Now concat')
        df_features = pd.concat(dfs_features)
        df_targets = pd.concat(dfs_targets)

        if (self.normalisation_mean is None) & (self.normalisation_std is None): 

            print ('Calculating normalisation parameters')
            #If we dont have any normalisation parameters already 
            self.normalisation_mean =  df_features.mean()
            self.normalisation_std =  df_features.std()

        #Normalise it using the pre calculated terms
        df_features = (df_features-self.normalisation_mean)/self.normalisation_std

        # Concat with the target variable which is unnormalised
        df_out = pd.concat([df_features,df_targets],axis=1)
        print(df_out)
        print(df_out.columns)

        #save it to disk
        print ('saving to',directory+'alldata.parquet')
        df_out.to_parquet(directory+'/alldata.parquet',compression=None)
   
       # return df_out 


    def greedy_preprocessing(self):

        print ('training data')
        print(self.normalisation_mean)
        self._process_directory(self.training_dir)  

        
        print ('validation data')
        print(self.normalisation_mean)
        self._process_directory(self.validation_dir) 
    