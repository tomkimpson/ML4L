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
    
    
    

def train_NN(x,y,
             x_val, y_val,
             epochs,batch_size,
             use_validation_data,pretrained_model):    
    """Train a sequential NN"""

    
    nfeatures = x.shape[-1]
    print('Training model:')
    print('Number of features:',nfeatures)
    print('Number of samples:',x.shape[0])
    print('Using validation data?', use_validation_data)
    print('Number of epochs:', epochs)
    print('Batch size:', batch_size)

    if pretrained_model is None:

        #Create a basic NN model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(int(nfeatures/2), activation='relu',input_shape=(nfeatures,),name='layer1'),
            tf.keras.layers.Dense(int(nfeatures/2), activation='relu',input_shape=(nfeatures,),name='layer2'),
            tf.keras.layers.Dense(1, name='output')
        ])

        #Compile it
        opt = tf.keras.optimizers.Adam(learning_rate=3e-4) 
        model.compile(optimizer=opt,
                    loss='mse',
                    metrics=['accuracy'])

    else:
        #Use a pretrained model and restart the training
        model = tf.keras.models.load_model(pretrained_model)


    #Early stop
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=50,
                                   verbose=1,
                                   mode='auto',
                                   baseline=None,
                                   restore_best_weights=True)
    
    #Checkpoints
    model_checkpoint = ModelCheckpoint(filepath = 'checkpoint', 
                                       monitor='val_loss', 
                                       save_best_only=True, 
                                       mode='min',
                                       save_freq='epoch',
                                       period=10)


    #Train it
    
    if use_validation_data:
        history = model.fit(x, y, 
                            epochs=epochs, batch_size=batch_size,
                            verbose=1,
                            validation_data=(x_val, y_val),
                            callbacks=[early_stopping,model_checkpoint]) 
    else:
        history = model.fit(x, y, 
                            epochs=epochs, batch_size=batch_size,
                            verbose=1) 

    return history,model
    

def save_model(output_path,model,history,parameters_dict):

    """Save model to disk after training """

    file_ID = str(uuid.uuid4().hex)
    fout = output_path+f'ML_{file_ID}/'
    
    #Create a directory
    os.makedirs(fout)
    print ('Writing data to:', fout)

    
    #Save the trained NN and the training history
    model.save(fout+'trained_model') 
    history_dict = history.history
    json.dump(history_dict, open(fout+'history.json', 'w'))
  
    
    #Save meta data as a txt file
    with open(fout+'meta.txt', 'w') as f:
        for k,v in parameters_dict.items():
            row = k + ": " + str(v) + "\n"
            f.write(row)

    return fout

            
            

#--------------------------------#
#----------Parameters------------#
#--------------------------------#
root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/'


#Inputs
training_data = root + 'joined_data/training_data.h5'
validation_data = root+ 'joined_data/validation_data.h5'

training_data = root + 'joined_data/training_data_with_monthly_lakes_w_lakes.h5'
validation_data = root+ 'joined_data/validation_data_with_monthly_lakes_w_lakes.h5'



#Model parameters
target_variable = ['MODIS_LST']   # The variable you are trying to learn/predict. Everything else is a model feature
do_not_use_delta_fields = False   # Don't use the V20 corrections
do_not_use_monthly_clakes = False # Don't use the monthly clake corrections
epochs = 100                      # Number of epochs for training
batch_size = 1024                 # Batch size for training
use_validation_data = True        # Do you want to use validation data for early stopping? Stopping conditions are defined in train_NN()
optimizer = 'adam'                #What optimiser to use







#Outputs
output_path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/processed_data/trained_models/'

#Use a pretrained model
#pretrained_model='/network/aopp/chaos/pred/kimpson/ML4L/reproducible_workflow/scripts/checkpoint'
pretrained_model = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/processed_data/trained_models/ML_5f2df91ccee94e63bef0ba4b5fc52cc3/trained_model'
#pretrained_model = None            

#--------------------------------#
#--------------MAIN--------------#
#--------------------------------#

#Get the matched data
print ('Reading training data')
df_train = pd.read_hdf(training_data)


print ('Reading validation data')
df_valid = pd.read_hdf(validation_data)



print ('Columns:')
print(df_train.columns)
print(df_valid.columns)


if do_not_use_delta_fields:
    print('Getting rid of delta fields')
    drop_columns = df_train.columns[df_train.columns.str.contains(pat = '_delta')]
    df_train = df_train.drop(drop_columns,axis=1)
    df_valid = df_valid.drop(drop_columns,axis=1)
    
if do_not_use_monthly_clakes:
    print('Getting rid of monthly clake fields')
    drop_columns = ['clake_monthly_value_delta']
    df_train = df_train.drop(drop_columns,axis=1)
    df_valid = df_valid.drop(drop_columns,axis=1)

print ('Total number of training samples:', len(df_train))
print ('Total number of validation samples:', len(df_valid))

# #Train model
print('Train the model')
history,model = train_NN(df_train.drop(target_variable,axis=1),df_train[target_variable],
                         df_valid.drop(target_variable,axis=1),df_valid[target_variable],
                         epochs,batch_size,use_validation_data,pretrained_model=pretrained_model)



print ('Model has completed training, now saving')

parameters_dict = {'training_data':   training_data,
                   'validation_data': validation_data,                  
                   'epochs':          epochs,
                   'batch_size':      batch_size,
                   'features':        list(df_train.drop(target_variable,axis=1).columns),
                   'optimizer':       optimizer}

fout = save_model(output_path,model,history,parameters_dict)


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
