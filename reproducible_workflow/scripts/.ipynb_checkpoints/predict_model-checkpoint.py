import tensorflow as tf
import os
import time
import json
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import xarray as xr
import uuid
import sys


"""
Script to train a sequential NN.
NN trains on training data, all results output to disk.
"""


### --- GPU configuration - dont use too much memory
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
           [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
### ---    

import ast 
def parse_meta_file(f):
    d = {}
    with open(f) as file:
        for line in file:
            key,val = line.rstrip().split(':')
            d[key] = val.strip() 

    features = d['features'].strip()

    return ast.literal_eval(features)



#--------------------------------#
#----------Parameters------------#
#--------------------------------#
root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/'






#Inputs
test_data =  root+'ECMWF_files/raw/processed_data/joined_data/test_data.h5'
#model_dir = root + 'processed_data/trained_models/ML_31c7f833c6f94de5a95d16bd9793b575/'
#model_dir = root + 'processed_data/trained_models/ML_bb1359c84c8845e5ac22185fc3686b96/'
model_dir = root + 'processed_data/trained_models/ML_5454e45e659043d6b295aac93aede77e/'


model = tf.keras.models.load_model(model_dir+'trained_model')
features = parse_meta_file(model_dir + 'meta.txt')
output_cols = ['latitude_ERA', 'longitude_ERA','time','skt','MODIS_LST'] #we don't output all columns in our results

print('features =', features)      
print('output cols = ',output_cols)

#--------------------------------#
#--------------MAIN--------------#
#--------------------------------#

#Get the matched data
print ('Reading test data')
df_test = pd.read_hdf(test_data)

print ('Columns:')
print(df_test.columns)







#Make some predictions
print(f'Using trained model {model_dir} to make some predictions')
predictions = model.predict(df_test[features])                 #Prediction 


#Create a selected results df
results_df = df_test[output_cols]
results_df['predictions'] = predictions                                      # Bring together the test data and predictions into a single pandas df
results_df.to_pickle(model_dir+'predictions.pkl')



print ("All completed OK")


















###---APPENDIX---###




### List of all features

# feature_names = ['sp', 'msl', 'u10', 'v10', 't2m',                                        # ERA_sfc, Time Variable
#                  'aluvp', 'aluvd', 'alnip', 'alnid', 'istl1', 'istl2', 'sd', 'd2m','fal', # ERA_skin, Time Variable
#                  'skt',                                                                   # ERA skt, Time Variable
#                  'lsm',  'slt', 'sdfor','lsrh', 'cvh',  'z', 'isor', 'sdor', 'cvl','cl','anor', 'slor', 'sr', 'tvh', 'tvl', #climatev15,v20 constant
#                  'vegdiff',                                                               #Bonus data
#                  'COPERNICUS/', 
#                  'CAMA/',
#                  'ORCHIDEE/', 
#                #  'monthlyWetlandAndSeasonalWater_minusRiceAllCorrected_waterConsistent/',
#                  'CL_ECMWFAndJRChistory/'
#                  ]


### List of all NEW renamed features


# feature_names = ['sp', 'msl', 'u10', 'v10', 't2m', 
#                  'aluvp', 'aluvd','alnip', 'alnid', 'istl1', 'istl2', 'sd', 'd2m', 'fal', 
#                  'skt',
#                  'slt_v15', 'sdfor_v15', 'vegdiff_v15', 'lsrh_v15', 'cvh_v15', 'lsm_v15','z_v15', 'isor_v15', 'sdor_v15', 'cvl_v15', 'cl_v15', 'anor_v15','slor_v15', 'sr_v15', 'tvh_v15', 'tvl_v15','dl_v15','si10_v15' 
#                  'slt_v20', 'sdfor_v20','vegdiff_v20', 'lsrh_v20', 'cvh_v20', 'lsm_v20', 'z_v20', 'isor_v20','sdor_v20', 'cvl_v20', 'cl_v20', 'anor_v20', 'slor_v20', 'sr_v20','tvh_v20', 'tvl_v20', 'dl_v20','si10_v20'
#                  'COPERNICUS/', 'CAMA/', 'ORCHIDEE/',
#                  #'monthlyWetlandAndSeasonalWater_minusRiceAllCorrected_waterConsistent/',
#                  'CL_ECMWFAndJRChistory/']





# #Calculate some extra features
#     df['cl_delta']  = df['cl_v20']  - df['cl_v15']
#     df['lsm_delta'] = df['lsm_v20'] - df['lsm_v15']
#     df['dl_delta']  = df['dl_v20']  - df['dl_v15']
#     df['cvh_delta'] = df['cvh_v20'] - df['cvh_v15']
#     df['cvl_delta'] = df['cvl_v20'] - df['cvl_v15']


#     df['anor_delta'] = df['anor_v20'] - df['anor_v15']
