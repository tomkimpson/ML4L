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

# def _bytes_feature(value):
#     """Returns a bytes_list from a string / byte."""
#     if isinstance(value, type(tf.constant(0))):
#         value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def _float_feature(value):
#     """Returns a float_list from a float / double."""
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# def _int64_feature(value):
#     """Returns an int64_list from a bool / enum / int / uint."""
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# def serialize_array(array):
#     array = tf.io.serialize_tensor(array)
#     return array



# #----------------------------------------------------------------------------------
# # Create example data
# array_blueprint = np.arange(4, dtype='float64').reshape(2,2)
# #arrays = [array_blueprint+1, array_blueprint+2, array_blueprint+3]

# new_blueprint = np.random.random(30).reshape(10,3).astype('float64')
# arrays = [new_blueprint]


data_files= natural_sort(glob.glob(f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/V2matched_*.pkl'))

training_files = data_files[0:12]
validation_files = data_files[12:24]
test_files = data_files[24:36]



def process_load(list_of_files,fname):

    print (fname)
    print('----------------------------------')
    dfs = []
    for f in list_of_files:
        print(f)
        df= pd.read_pickle(f)



        
 
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


data_files= natural_sort(glob.glob(f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/V2matched_*.pkl'))

training_files = data_files[0:12]
validation_files = data_files[12:24]
test_files = data_files[24:36]



def process_load(list_of_files,fname):

    print (fname)
    print('----------------------------------')
    dfs = []
    for f in list_of_files:
        print(f)
        df= pd.read_pickle(f)
        dfs.append(df)
        
        
    print('Concat')
    df = pd.concat(dfs)
    
    
    print ('Writing CSV')
    fout = f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/{fname}.csv'
    df.to_csv(fout)       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


#see https://stackoverflow.com/questions/47861084/how-to-store-numpy-arrays-as-tfrecord

#----------------------------------------------------------------------------------
# # Write TFrecord file
# file_path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/data.tfrecords'
# with tf.io.TFRecordWriter(file_path) as writer:
#     for array in arrays:
#         print(array)
#         serialized_array = serialize_array(array)
#         feature = {'b_feature': _bytes_feature(serialized_array)}
#         example_message = tf.train.Example(features=tf.train.Features(feature=feature))
#         writer.write(example_message.SerializeToString())

        
        
        
        
# def write_record():
#     # Read image raw data, which will be embedded in the record file later.
#     image_string = open('image.jpg', 'rb').read()
    
#     # Manually set the label to 0. This should be set according to your situation.
#     label = 0
    
#     # For each sample there are two features: image raw data, and label. Wrap them in a single dict.
#     feature = {
#         'label': _int64_feature(label),
#         'image_raw': _bytes_feature(image_string),
#     }
    
#     # Create a `example` from the feature dict.
  
#     # Write the serialized example to a record file.
#     with tf.io.TFRecordWriter(file_path) as writer:
        
        
#         feature = {'image': _bytes_feature(serialized_featurues_array),
#                    'label':  _bytes_feature(serialized_featurues_array)}
        
        
#         tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

        
        
        
#         writer.write(tf_example.SerializeToString())  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

# # Read TFRecord file
# def _parse_tfr_element(element):
#     parse_dic = {
#         'b_feature': tf.io.FixedLenFeature([], tf.string), # Note that it is tf.string, not tf.float32
#             }
#     example_message = tf.io.parse_single_example(element, parse_dic)

#     b_feature = example_message['b_feature'] # get byte string
#     feature = tf.io.parse_tensor(b_feature, out_type=tf.float64) # restore 2D array from byte string
#     return feature




# tfr_dataset = tf.data.TFRecordDataset(file_path) 
# dataset = tfr_dataset.map(_parse_tfr_element)
# for instance in dataset:
#     print(instance)

# def read_tfrecord(example, labeled):
#     tfrecord_format = (
#         {
#             "image": tf.io.FixedLenFeature([], tf.string),
#             "target": tf.io.FixedLenFeature([], tf.int64),
#         }
#         if labeled
#         else {"image": tf.io.FixedLenFeature([], tf.string),}
#     )
#     example = tf.io.parse_single_example(example, tfrecord_format)
#     image = decode_image(example["image"])
#     if labeled:
#         label = tf.cast(example["target"], tf.int32)
#         return image, label
#     return image



# tfr_dataset = tf.data.TFRecordDataset(file_path) 
# for serialized_instance in tfr_dataset:
#     print(serialized_instance) # print serialized example messages

# dataset = tfr_dataset.map(_parse_tfr_element)
# for instance in dataset:
#     print()
#     print(instance) # print parsed example messages with restored arrays
































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



