# import re
# import glob
# import xarray as xr
# import pandas as pd
# import sys
# import numpy as np

#https://www.tensorflow.org/tutorials/load_data/tfrecord#reading_a_tfrecord_file

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


def serialize_example(feature0, feature1, feature2):
  
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
      'feature0': _float_feature(feature0),
      'feature1': _float_feature(feature1),
      'feature2': _float_feature(feature2)
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(f0,f1,f2):
    tf_string = tf.py_function(
                               serialize_example,
                               (f0, f1, f2),  # Pass these args to the above function.
                               tf.string)      # The return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar.

#----------------------------------------------------------------------------------
# Create example data

f = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/training_data/V2matched_0.pkl'
example_file =pd.read_pickle(f)

feature0 = example_file['MODIS_LST']
feature1 = example_file['t2m']
feature2 = example_file['sp']

#Make it a TF dataset
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2))

#print
for f0,f1,f2 in features_dataset.take(1): #use `take(1)` to pull a single example from the dataset
    
#apply function to each element in the dataset
serialized_features_dataset = features_dataset.map(tf_serialize_example)


#write it to disk
print('Writing to disk')
filename = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/test_data.tfrecords'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)







#READ IT
print('Now reading')
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
#This raw dataset contains serialised tf.train.Example messages. When iterated over it returns these as scalar string tensor, e.g.


for raw_record in raw_dataset.take(10):
    print(repr(raw_record))












#----------------------------------------------------------------------------------
# # Write TFrecord file
# file_path = 
# with tf.io.TFRecordWriter(file_path) as writer:
#     for array in arrays:
#         print(array)
#         serialized_array = serialize_array(array)
#         feature = {'b_feature': _bytes_feature(serialized_array)}
#         example_message = tf.train.Example(features=tf.train.Features(feature=feature))
#         writer.write(example_message.SerializeToString())









        
        
        
#feature = _float_feature(np.exp(1))
#feature.SerializeToString()












