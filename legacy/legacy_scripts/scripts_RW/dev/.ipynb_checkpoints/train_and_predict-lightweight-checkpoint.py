import tensorflow as tf
import os
import time
import json
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import xarray as xr
import uuid
import sys



#Temporary region
#method_type = sys.argv[1]

#calculate_delta_fields = sys.argv[1]



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
    
    
    

    
def train_NN(x,y,epochs,batch_size,optimizer):    
    """Train a sequential NN"""

    
    nfeatures = x.shape[-1]
    print('Training model:')
    print('Number of features:',nfeatures)
    print('Number of samples:',x.shape[0])
    
    # print ('Creating a normalization layer')
    # normalizer = tf.keras.layers.Normalization(axis=-1)
    # normalizer.adapt(x)
    

    #Create a basic NN model
    model = tf.keras.Sequential([
       # normalizer,
        tf.keras.layers.Dense(int(nfeatures/2), activation='relu',input_shape=(nfeatures,),name='layer1'),
        tf.keras.layers.Dense(int(nfeatures/2), activation='relu',input_shape=(nfeatures,),name='layer2'),
        tf.keras.layers.Dense(1, name='output')
      ])

    #Compile it
    print('compile')
    opt = tf.keras.optimizers.Adam()#(learning_rate=3e-4) 
    model.compile(optimizer=opt,
                  loss='mse',
                  metrics=['accuracy'])
    

 
    
    #Run it
    print('GO')
    history = model.fit(x, y, 
                        epochs=epochs, batch_size=batch_size,
                        verbose=1) 

    return history,model
    


           


#--------------------------------#
#----------Parameters------------#
#--------------------------------#
root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/'



#Network parameters
epochs = 1000
batch_size = 1024
use_validation_data = True #Do you want to use validation data for early stopping? Stopping conditions are defined in train_NN()
optimizer = 'adam'


                 
 

training_data = pd.read_pickle('/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/example_training_data.pkl')
y = training_data.pop('MODIS_LST')


train_NN(training_data,y,epochs,batch_size,optimizer)

# print('Feature names:')
# print(feature_names)


     
# #print(df.isna().any())

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



# print ('Save the trained model')

# parameters_dict = {'input_file':     input_file,
#                   'train_condition': train_condition,
#                   'test_condition':  test_condition,
#                   'epochs':          epochs,
#                   'batch_size':      batch_size,
#                   'features':        feature_names,
#                   'optimizer':       optimizer}
# fout = save_model(output_path,model,history,parameters_dict)


# #Make some predictions
# print('Predict')
# predictions_normalized = model.predict(split_data['x_test'])                 #Prediction 
# predictions = (predictions_normalized * T_normalize_std ) + T_normalize_mean #un-normalsie to get 'physical' value
# results_df['predictions'] = predictions                                      # Bring together the test data and predictions into a single pandas df
# results_df.to_pickle(fout+'predictions.pkl')



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
