# import re
# import glob
# import xarray as xr
# import pandas as pd
# import sys
# import numpy as np



"""
Script that takes all the month files output bu join_MODIS_with-ERA.py and unifies
them into a single HDF.
"""

import tensorflow as tf
import numpy as np
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array



#----------------------------------------------------------------------------------
# Create example data
array_blueprint = np.arange(4, dtype='float64').reshape(2,2)
arrays = [array_blueprint+1, array_blueprint+2, array_blueprint+3]

#----------------------------------------------------------------------------------
# Write TFrecord file
file_path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/data.tfrecords'
with tf.io.TFRecordWriter(file_path) as writer:
    for array in arrays:
        print(array)
        serialized_array = serialize_array(array)
        feature = {'b_feature': _bytes_feature(serialized_array)}
        example_message = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example_message.SerializeToString())





















