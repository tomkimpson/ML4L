'''
Script to turn our monthly joined ERA-MODIS *.pkl files to a TFRecord format.
See https://www.tensorflow.org/tutorials/load_data/tfrecord#reading_a_tfrecord_file
'''

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
    
    subdir = f.split('/')[-2] 
    f1 = f.split('/')[-1].split('.pkl')[0]
    fname = f'{f1}.tfrecords'
    
    output_file = f'{root}{subdir}/TFRecords/{fname}'
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
    
    print(df.shape)
    sys.exit()
    with tf.io.TFRecordWriter(output_filename) as writer:
        
        print (f'Writing {len(df)} rows to {output_filename}')
        for i in range(len(df)):
            input_data = df.values[i] #features
            l = [target.values[i]]    #targets
            writer.write(serialize_example(input_data,l))
        
def process_directory(directory):
    print(directory)
    input_files = glob.glob(f'{root}{directory}/*.pkl')
    for f in input_files:
        convert_month_to_TFRecord(f)


process_directory('training_data_with_monthly_lakes')   #Training data
process_directory('validation_data_with_monthly_lakes') #Validation data


















