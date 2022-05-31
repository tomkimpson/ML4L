# -*- coding: utf-8 -*-
"""Data Loader"""

import tensorflow as tf
import pandas as pd
import os
import json
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
    def load_training_data(data_config):
        """
        Loads dataset from path.
        Only columns specified in training_features, target_variable are loaded
        """
        

        print ('Loading training data from file:', data_config.training_data)
        print ('Loading validation data from file:', data_config.validation_data)

        training_data_size = os.path.getsize(data_config.training_data)
        validation_data_size = os.path.getsize(data_config.validation_data)

        print ('Size of training data:', round(training_data_size/1e9,2) , ' G')
        print ('Size of validation data:', round(validation_data_size/1e9,2) , ' G')

        #Only load the training features and the target variable
        cols = data_config.training_features + [data_config.target_variable]
        return pd.read_parquet(data_config.training_data,columns=cols), pd.read_parquet(data_config.validation_data,columns=cols) 


    @staticmethod
    def load_testing_data(data_config):
        """
        Loads dataset from path.
        Only columns specified in training_features, target_variable are loaded
        """
        
        model_load_dir = data_config.train.path_to_trained_models + data_config.train.model_name #Where the trained model is

        with open(model_load_dir+'/configuration.json') as f:
            config_tmp=json.load(f)
            columns_used_by_model = config_tmp['train']['training_features'] 


        test_data = pd.read_parquet(data_config.predict.testing_data,columns=columns_used_by_model + [data_config.train.target_variable] )
        
        test_data_size = os.path.getsize(data_config.predict.testing_data)
        print ('Loading test data from file:', data_config.predict.testing_data)
        print ('Size of test data:', round(test_data_size/1e9,2) , ' G')


        try:
            test_data = test_data.query(data_config.predict.testing_data_query)
            print('Selecting a subset of test data according to query:',data_config.predict.testing_data_query)
        except:
            test_data = test_data

        #Only load the training features and the target variable
        return columns_used_by_model, test_data
