# import re
# import glob
# import xarray as xr
# import pandas as pd
# import sys
# import numpy as np

#https://www.tensorflow.org/tutorials/load_data/tfrecord#reading_a_tfrecord_file


import tensorflow as tf
import numpy as np
import pandas as pd
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





def tf_serialize_example(f0,f1,f2):
    tf_string = tf.py_function(
                               serialize_example,
                               (f0, f1, f2),  # Pass these args to the above function.
                               tf.string)      # The return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar.

#----------------------------------------------------------------------------------
# Create example data

print ('Creating example data')
# The number of observations in the dataset.
n_observations = int(1e4)

feature0 = np.random.randn(n_observations)

feature1 = np.random.randn(n_observations)

feature2 = np.random.randn(n_observations)

feature3 = np.random.randn(n_observations)

print('Creating dataset from these features')

#Create a dataset of these features
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))


# Use `take(1)` to only pull one example from the dataset.
for f0,f1,f2,f3 in features_dataset.take(1):
    print(f0)
    print(f1)
    print(f2)
    print(f3)
    
    
    
    
    
def serialize_example(feature0, feature1, feature2, feature3):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()





#A non tensor function like serialize_example needs to be wrapped  to make it compatible
def tf_serialize_example(f0,f1,f2,f3):
    tf_string = tf.py_function(
                             serialize_example,
                             (f0, f1, f2, f3),  # Pass these args to the above function.
                              tf.string
                             )      # The return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar.




#Serialise it all, wrapped 
print('serialising')
serialized_features_dataset = features_dataset.map(tf_serialize_example)

print(serialized_features_dataset)




def generator():
    for features in features_dataset:
        yield serialize_example(*features)




serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=())





output_filename = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/test_data.tfrecords'
writer = tf.data.experimental.TFRecordWriter(output_filename)
writer.write(serialized_features_dataset)





#----------------------------------------------------------------------------------







# print ('Loading data')

# f = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/training_data/V2matched_0.pkl'
# example_file =pd.read_pickle(f)

# feature0 = example_file['MODIS_LST']
# feature1 = example_file['t2m']
# feature2 = example_file['sp']

# #Make it a TF dataset
# features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2))

# #print
# for f0,f1,f2 in features_dataset.take(1): #use `take(1)` to pull a single example from the dataset
#     print(f0)
#     print(f1)
#     print(f2)
# #apply function to each element in the dataset
# serialized_features_dataset = features_dataset.map(tf_serialize_example)


# #write it to disk
# print('Writing to disk')
# filename = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/test_data.tfrecords'
# writer = tf.data.experimental.TFRecordWriter(filename)
# #writer = tf.python_io.TFRecordWriter(result_tf_file)

# writer.write(serialized_features_dataset)








# # Create a description of the features.
# feature_description = {
#     'feature0': tf.io.FixedLenFeature([], tf.int32, default_value=0),
#     'feature1': tf.io.FixedLenFeature([], tf.int32, default_value=0),
#     'feature2': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
# }

# def _parse_function(example_proto):
#   # Parse the input `tf.train.Example` proto using the dictionary above.
#   return tf.io.parse_single_example(example_proto, feature_description)


# parsed_dataset = raw_dataset.map(_parse_function)







# for parsed_record in parsed_dataset.take(10):
#   print(repr(parsed_record))



#    score = tf.to_float(features[F_SCORE])
#     votes = features[F_VOTES]
#     helpfulness = features[F_HELPFULNESS]







# def get_batched_dataset(filenames):
#   option_no_order = tf.data.Options()
#   option_no_order.experimental_deterministic = False

#   dataset = tf.data.Dataset.list_files(filenames)
#   dataset = dataset.with_options(option_no_order)
#   dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
#   dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

#   dataset = dataset.cache() # This dataset fits in RAM
#   dataset = dataset.repeat()
#   dataset = dataset.shuffle(2048)
#   dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) 
#   dataset = dataset.prefetch(AUTO) #
  
#   return dataset
















