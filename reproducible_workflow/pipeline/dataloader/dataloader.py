# -*- coding: utf-8 -*-
"""Data Loader"""

import tensorflow as tf

class DataLoader:
    """Data Loader class"""    


    

    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""
        return tf.data.TFRecordDataset(data_config.training_data),tf.data.TFRecordDataset(data_config.validation_data)
    
