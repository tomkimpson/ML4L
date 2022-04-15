import tensorflow as tf
import os
import time
import json
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import xarray as xr
import uuid



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
    
    
import tensorflow_io as tfio
    
    
#--------------------------------#
#----------Parameters------------#
#--------------------------------#
root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/'

#Inputs
version = 'v15' #v20
input_file = f'{root}processed_data/joined_data/{version}/all_months.h5'
print('Getting ds')
ds = tfio.IODataset.from_hdf5(input_file)


print ('got ds')
            
            
            





# #Outputs
# output_path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/processed_data/trained_models/'
# output_cols = ['latitude_ERA', 'longitude_ERA','time','skt','MODIS_LST'] #we don't output all columns in our results

# #Train/Test split. Everything not in these two groups is validation data.
# train_condition = pd.to_datetime("2019-01-01 00:00:00") #Everything less than this is used for training data
# test_condition  = pd.to_datetime("2020-01-01 00:00:00") #Everything greater than this is used for test data. 
# feature_names = ['sp', 'msl', 'u10', 'v10', 't2m',
#             'aluvp', 'aluvd', 'alnip', 'alnid', 'istl1', 'istl2', 'sd', 'd2m','fal', 
#             'skt', 
#             'lsm',  'slt', 'sdfor','lsrh', 'cvh',  'z', 'isor', 'sdor', 'cvl','cl','anor', 'slor', 'sr', 'tvh', 'tvl']

# target_var = ['MODIS_LST'] 


# #Model parameters
# epochs = 100
# batch_size = 100000
# use_validation_data = True #Do you want to use validation data for early stopping? Stopping conditions are defined in train_NN()
# optimizer = 'adam'

                 
            

# #--------------------------------#
# #--------------MAIN--------------#
# #--------------------------------#

# #Get the matched data
# print ('Reading data')
# df= pd.read_hdf(input_file)
    
# #Normalise everything
# print('Normalizing')
# features = df[feature_names]
# target = df[target_var]
    
# features_normed = (features-features.mean())/features.std()
# target_normed = (target-target.mean())/target.std()


# T_normalize_mean = target.mean().values[0] #Save these as np.float32, rather than pd.Series. We will use them later to "un-normalize"
# T_normalize_std = target.std().values[0]

# #Split data into training and testing set
# print('Split data')
# split_data,results_df = train_test_split(df,train_condition,test_condition,
#                                                             features_normed,target_normed,
#                                                             output_cols)


# #Train model
# print('Train')
# history,model = train_NN(split_data['x_train'],split_data['y_train'],
#                          split_data['x_valid'],split_data['y_valid'],
#                          epochs,batch_size,use_validation_data,optimizer)


# #Make some predictions
# print('Predict')
# predictions_normalized = model.predict(split_data['x_test'])


# #predictions = (predictions_normalized * target.std() ) + target.mean() #un-normalsie to get 'physical' value
# predictions = (predictions_normalized * T_normalize_std ) + T_normalize_mean #un-normalsie to get 'physical' value


# #Bring together the test data and predictions into a single pandas df
# results_df['predictions'] = predictions



# #Save everything to disk
# parameters_dict = {'input_file':     input_file,
#                   'train_condition': train_condition,
#                   'test_condition':  test_condition,
#                   'epochs':          epochs,
#                   'batch_size':      batch_size}
                                   


# write_outputs(output_path,model,history,results_df,parameters_dict)

    


# print ("All completed OK")
