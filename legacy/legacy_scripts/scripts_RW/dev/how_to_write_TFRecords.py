#https://www.tensorflow.org/tutorials/load_data/tfrecord#reading_a_tfrecord_file


import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import re
import sys

root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/'



# Index([
#        'MODIS_LST', 'sp', 'msl', 'u10', 'v10', 't2m', 'aluvp', 'aluvd',
#        'alnip', 'alnid', 'istl1', 'istl2', 'sd', 'd2m', 'fal', 'skt',
#        'slt_v15', 'sdfor_v15', 'vegdiff_v15', 'lsrh_v15', 'cvh_v15',
#        'isor_v15', 'dl_v15', 'lsm_v15', 'z_v15', 'si10_v15', 'sdor_v15',
#        'cvl_v15', 'anor_v15', 'slor_v15', 'sr_v15', 'tvh_v15', 'tvl_v15',
#        'cl_v15', 'slt_v20', 'sdfor_v20', 'vegdiff_v20', 'lsrh_v20', 'cvh_v20',
#        'isor_v20', 'dl_v20', 'lsm_v20', 'z_v20', 'si10_v20', 'sdor_v20',
#        'cvl_v20', 'anor_v20', 'slor_v20', 'sr_v20', 'tvh_v20', 'tvl_v20',
#        'cl_v20', 'clake_monthly_value', 'heightAboveGround', 'L2_distance',
#        'H_distance', 'time'],


#Drop these columns. They are not needed for training.
#Note that all the V20 fields will also be dropped.
#We retain instead the "delta fields" (V20-V15) for the undropped V15 features
drop_cols = ['latitude_ERA', 'longitude_ERA', 'latitude_MODIS', 'longitude_MODIS', #locations
             'slt_v15', 'sdfor_v15', 'vegdiff_v15','z_v15','tvh_v15', 'tvl_v15',   #Dont use these V15 fields for now
             'heightAboveGround', 'L2_distance','H_distance', 'time']

#Initialise normalisation factors.
#These will be computed once for the first batch (i.e. month) loaded and then applied to all subsequent months
normalisation_mean = None
normalisation_std = None 


def serialize_example(input_data,label):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
      'feature': tf.train.Feature(float_list=tf.train.FloatList(value=input_data)),
      'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
            }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()




def parse_filename(f):
      
    """Map an input filename to an output filename"""
    
    f1 =f.split('training_data_with_monthly_lakes/')[-1].split('.pkl')[0] #Ugly. Must be a nicer way to do this
    fname = f'{f1}.tfrecords'
    
    output_file = f'{root}training_data_with_monthly_lakes/TFRecords/{fname}'
    return output_file    

def calculate_V20_corrections(df):
    
    """For the time constant fields determine V20-V15. """
    time_constant_features = [col for col in df if col.endswith('_v15')]
    for i in time_constant_features:
        feature = i.split('_')[0]        # e.g. cl_v15 --> cl
        column_name = f'{feature}_delta' # e.g. cl     --> cl_delta
        v20_name = feature+'_v20'        # e.g. cl_v20
    
        df[column_name] = df[v20_name] - df[i]
    
    
    return df


def normalize_features(df):
    
    """Normalize all the features using the global
       parameters normalisation_mean and normalisation_std"""

    global normalisation_mean
    global normalisation_std
    
    if (normalisation_mean is None) & (normalisation_std is None):
        # For the first batch only, calculate the normalisation parameters
        print ('Calculating the normalization parameters for the first batch')
        #If we dont have any normalisation parameters already 
        normalisation_mean =  df.mean()
        normalisation_std =  df.std()
        
        #Write them to disk
        normalisation_mean.to_pickle('normalisation_mean.pkl')
        normalisation_std.to_pickle('normalisation_std.pkl')


    #Normalise it using the pre calculated terms
    return (df-normalisation_mean)/normalisation_std
    

def convert_month_to_TFRecord(f):
    
    print('Converting monthly file: ', f)

    df = pd.read_pickle(f)              # Load monthly file
    df = df.drop(columns=drop_cols)     # Drop the columns that we don't want to pass to model training
    target = df.pop('MODIS_LST')        # Pop out target/label/output column
    df = calculate_V20_corrections(df)  # Get V20-V15 features
    df = df.drop(columns=[col for col in df if col.endswith('_v20')])  # Now drop the leftover V20 fields
    df = normalize_features(df)         # Normalize the features, not the target.
    output_filename = parse_filename(f) # Name the output file based on the input file
    
    
    with tf.io.TFRecordWriter(output_filename) as writer:
        
        print (f'Writing {len(df)} rows to {output_filename}')
        for i in range(len(df)):
            input_data = df.values[i] #features
            l = [target.values[i]]    #targets
            writer.write(serialize_example(input_data,l))
        
def process_directory(directory):
    input_files = glob.glob(f'{root}{directory}/*.pkl')
    for f in input_files:
        convert_month_to_TFRecord(f)


process_directory('training_data_with_monthly_lakes')   #Training data
process_directory('validation_data_with_monthly_lakes') #Validation data

















#-------------OLD MATERIAL----------------


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

# def _float_array_feature(value):
#     """Returns a float_list from an array of float / double."""
#     return tf.train.Feature(float_list=tf.train.FloatList(value=value))





# def tf_serialize_example(f0,f1,f2):
#     tf_string = tf.py_function(
#                                serialize_example,
#                                (f0, f1, f2),  # Pass these args to the above function.
#                                tf.string)      # The return type is `tf.string`.
#     return tf.reshape(tf_string, ()) # The result is a scalar.

#----------------------------------------------------------------------------------
# Create example data


# #A non tensor function like serialize_example needs to be wrapped  to make it compatible
# def tf_serialize_example(f0,f1,f2,f3):
#     tf_string = tf.py_function(
#                              serialize_example,
#                              (f0, f1, f2, f3),  # Pass these args to the above function.
#                               tf.string
#                              )      # The return type is `tf.string`.
#     return tf.reshape(tf_string, ()) # The result is a scalar.




# #Serialise it all, wrapped 
# print('serialising')
# serialized_features_dataset = features_dataset.map(tf_serialize_example)

# print(serialized_features_dataset)




# def generator():
#     for features in features_dataset:
#         yield serialize_example(*features)




# serialized_features_dataset = tf.data.Dataset.from_generator(
#     generator, output_types=tf.string, output_shapes=())





# output_filename = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/test_data.tfrecords'
# writer = tf.data.experimental.TFRecordWriter(output_filename)
# writer.write(serialized_features_dataset)





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
















