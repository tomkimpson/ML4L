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

































# def natural_sort(l): 
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
#     return sorted(l, key=alphanum_key)


# data_files= natural_sort(glob.glob(f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/V2matched_*.pkl'))

# training_files = data_files[0:12]
# validation_files = data_files[12:24]
# test_files = data_files[24:36]



# def process_load(list_of_files,fname):

#     print (fname)
#     print('----------------------------------')
#     dfs = []
#     for f in list_of_files:
#         print(f)
#         df= pd.read_pickle(f)
#         dfs.append(df)
        
        
#     print('Concat')
#     df = pd.concat(dfs)
    
    
#     print ('Writing CSV')
#     fout = f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/{fname}.csv'
#     df.to_csv(fout)


# process_load(training_files,'training_data')
# process_load(training_files,'validation_data')
# process_load(training_files,'testing_data')
    




# train_valid_test = {'train': 12,
#                    'valid':}



# dfs = []
# for f in data_files[0:1]:
#     print(f)
#     df= pd.read_pickle(f)
#     dfs.append(df)

# print('Concat')
# df = pd.concat(dfs)
# print(df.columns)

# #print ('Writing NPY')
# #fout = f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/test_file.npy'
# #npy_file = df.to_numpy()
# #np.save(fout,npy_file)

# print ('Writing CSV')
# fout = f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/test_file.csv'
# df.to_csv(fout)


#print('Writing HDF')
#fout = f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/all_months_V3.h5'
#df.to_hdf(fout, key='df', mode='w') 
#print('Done')















# for v in ['v15', 'v20']:
#     data_files= natural_sort(glob.glob(f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/{v}/matched_*.pkl'))
#     print('v:', v)
#     dfs = []
#     for f in data_files:
#         print(f)
#         df= pd.read_pickle(f)
#         dfs.append(df)
    
#     print('Concat')
#     df = pd.concat(dfs)

#     print('Writing HDF')
#     fout = f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/{v}/all_months.h5'
#     df.to_hdf(fout, key='df', mode='w') 
#     print('Done')



