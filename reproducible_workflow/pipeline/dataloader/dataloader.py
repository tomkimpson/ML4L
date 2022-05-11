# -*- coding: utf-8 -*-
"""Data Loader"""

import tensorflow as tf
import pandas as pd
import os
class DataLoader:
    """Data Loader class"""    


    

    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""

        training_data_size = os.path.getsize(data_config.training_data)
        validation_data_size = os.path.getsize(data_config.validation_data)

        print ('Size of training data:', training_data_size/1e9 , ' G')
        print ('Size of validation data:', validation_data_size/1e9 , ' G')


        #return tf.data.TFRecordDataset(data_config.training_data),tf.data.TFRecordDataset(data_config.validation_data)
        return pd.read_hdf(data_config.training_data), pd.read_hdf(data_config.validation_data), 
