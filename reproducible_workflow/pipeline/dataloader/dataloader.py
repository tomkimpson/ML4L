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

        s1 = os.path.getsize(data_config.training_data)
        s2 = os.path.getsize(data_config.validation_data)

        print ('size s1:', s1)

        #return tf.data.TFRecordDataset(data_config.training_data),tf.data.TFRecordDataset(data_config.validation_data)
        return pd.read_hdf(data_config.training_data), pd.read_hdf(data_config.validation_data), 
