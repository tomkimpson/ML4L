# -*- coding: utf-8 -*-
"""Data Loader"""

import tensorflow as tf
import pandas as pd
import os
class DataLoader:
    """Data Loader class"""    


    

    # @staticmethod
    # def load_data(data_config):
    #     """Loads dataset from path"""

    #     training_data_size = os.path.getsize(data_config.training_data)
    #     validation_data_size = os.path.getsize(data_config.validation_data)

    #     print ('Size of training data:', round(training_data_size/1e9,2) , ' G')
    #     print ('Size of validation data:', round(validation_data_size/1e9,2) , ' G')


    #     #return tf.data.TFRecordDataset(data_config.training_data),tf.data.TFRecordDataset(data_config.validation_data)
    #     return pd.read_hdf(data_config.training_data), pd.read_hdf(data_config.validation_data), 

    

    @staticmethod
    def load_parquet_data(data_config):
        """Loads dataset from path"""

        print ('Loading training data from file:', data_config.training_data)
        print ('Loading validation data from file:', data_config.validation_data)

        training_data_size = os.path.getsize(data_config.training_data)
        validation_data_size = os.path.getsize(data_config.validation_data)

        print ('Size of training data:', round(training_data_size/1e9,2) , ' G')
        print ('Size of validation data:', round(validation_data_size/1e9,2) , ' G')

        #Only load the training features and the target variable
        return pd.read_parquet(data_config.training_data,columns=data_config.training_features + [data_config.target_variable]), pd.read_parquet(data_config.validation_data,columns=data_config.training_features+ [data_config.target_variable]) 
