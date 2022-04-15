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
    
    
    
    
### --- Define all functions
def train_test_split(df,
                     train_condition,test_condition,
                     normalised_features,normalised_targets,
                     output_columns):
    
   
    #Separate into train/test based on time
    idx_train = df.time < train_condition 
    idx_valid = (train_condition <= df.time) & (df.time <= test_condition) 
    idx_test = df.time > test_condition 
    
    print ('idx valid = ', idx_valid.sum())
    
    split_data = {'x_train': normalised_features[idx_train],
                  'y_train': normalised_targets[idx_train],
                  'x_valid': normalised_features[idx_valid], 
                  'y_valid': normalised_targets[idx_valid],
                  'x_test' : normalised_features[idx_test],
                  'y_test' : normalised_targets[idx_test]
                 }
    
    # Create a "results_df" copy that will be used later for smaller IO. Won't output all features
    results_df = df[output_columns].copy()
    results_df = results_df[idx_test] #For IO we will just output predictions
    
    
    return split_data,results_df
    
    
    
    

def train_NN(x,y,x_val, y_val,epochs,batch_size,use_validation_data,optimizer):    
    """Train a sequential NN"""

    
    nfeatures = x.shape[-1]
    print('Training model',nfeatures)

    #Create a basic NN model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(int(nfeatures/2), activation='relu',input_shape=(nfeatures,),name='layer1'),
        tf.keras.layers.Dense(1, name='output')
      ])

    #Compile it
    
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['accuracy'])
    

    
    #Early stop
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=10,
                                   verbose=1,
                                   mode='auto',
                                   baseline=None,
                                   restore_best_weights=True)
    
    #Train it
    
    if use_validation_data:
        history = model.fit(x, y, 
                            epochs=epochs, batch_size=batch_size,
                            verbose=1,
                            validation_data=(x_val, y_val),
                            callbacks=[early_stopping]) 
    else:
        history = model.fit(x, y, 
                            epochs=epochs, batch_size=batch_size,
                            verbose=1) 

    return history,model
    

def save_model(output_path,model,history,df,parameters_dict):

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
root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/'

#Inputs
version = 'v20' #v20
input_file = f'{root}processed_data/joined_data/{version}/all_months.h5'


#Outputs
output_path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/processed_data/trained_models/'
output_cols = ['latitude_ERA', 'longitude_ERA','time','skt','MODIS_LST'] #we don't output all columns in our results

#Train/Test split. Everything not in these two groups is validation data.
train_condition = pd.to_datetime("2019-01-01 00:00:00") #Everything less than this is used for training data
test_condition  = pd.to_datetime("2020-01-01 00:00:00") #Everything greater than this is used for test data. 
feature_names = ['sp', 'msl', 'u10', 'v10', 't2m',
            'aluvp', 'aluvd', 'alnip', 'alnid', 'istl1', 'istl2', 'sd', 'd2m','fal', 
            'skt', 
            'lsm',  'slt', 'sdfor','lsrh', 'cvh',  'z', 'isor', 'sdor', 'cvl','cl','anor', 'slor', 'sr', 'tvh', 'tvl']

# feature_names = [ 'sp', 'msl', 'u10', 'v10','t2m',
#                          'aluvp', 'aluvd', 'alnip', 'alnid', 'cl',
#                          'cvl', 'cvh', 'slt', 'sdfor', 'z', 'sd', 'sdor', 'isor', 'anor', 'slor',
#                          'd2m', 'lsm', 'fal','skt'] 




target_var = ['MODIS_LST'] 


#Model parameters
epochs = 200
batch_size = 1024
use_validation_data = False #Do you want to use validation data for early stopping? Stopping conditions are defined in train_NN()
optimizer = 'adam'

parameters_dict = {'input_file':     input_file,
                  'train_condition': train_condition,
                  'test_condition':  test_condition,
                  'epochs':          epochs,
                  'batch_size':      batch_size}
                 
            

#--------------------------------#
#--------------MAIN--------------#
#--------------------------------#

#Get the matched data
print ('Reading data')
df= pd.read_hdf(input_file)
    
#Normalise everything
print('Normalizing')
features = df[feature_names]
target = df[target_var]
    
features_normed = (features-features.mean())/features.std()
target_normed = (target-target.mean())/target.std()


T_normalize_mean = target.mean().values[0] #Save these as np.float32, rather than pd.Series. We will use them later to "un-normalize"
T_normalize_std = target.std().values[0]

#Split data into training and testing set
print('Split data')
split_data,results_df = train_test_split(df,train_condition,test_condition,
                                                            features_normed,target_normed,
                                                            output_cols)


#Train model
print('Train')
history,model = train_NN(split_data['x_train'],split_data['y_train'],
                         split_data['x_valid'],split_data['y_valid'],
                         epochs,batch_size,use_validation_data,optimizer)



print ('Save the trained model')
fout = save_model(output_path,model,history,parameters_dict)


#Make some predictions
print('Predict')
predictions_normalized = model.predict(split_data['x_test'])                 #Prediction 
predictions = (predictions_normalized * T_normalize_std ) + T_normalize_mean #un-normalsie to get 'physical' value
results_df['predictions'] = predictions                                      # Bring together the test data and predictions into a single pandas df
results_df.to_pickle(fout+'predictions.pkl')



print ("All completed OK")
