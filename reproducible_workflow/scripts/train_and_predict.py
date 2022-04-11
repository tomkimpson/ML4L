import tensorflow as tf
import os
import time
import json
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import xarray as xr

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

    print('Training model')
    nfeatures = x.shape[-1]


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
    



def write_outputs(output_path,model,history,df,parameters_dict):
    """Write all outputs to disk"""
    
    
    
    
    ID = f'NNModel_nfeatures{str(len(feature_names))}_nepochs{str(epochs)}_bs{str(batch_size)}_{str(int(time.time()))}/'
print(ID)
    
    
    
    
    #Create a directory
    #id = int(time.time())
    #fout = output_path+f'ML_{str(id)}/'
    os.makedirs(fout)
    print ('Writing data to:', fout)

    
    #Save the trained NN and the training history
    model.save(fout+'trained_model') 
    history_dict = history.history
    json.dump(history_dict, open(fout+'history.json', 'w'))
    
    #Save the prediction results
    df.to_pickle(fout+'predictions.pkl')
    
    
    #Save meta data as a txt file
    with open(fout+'meta.txt', 'w') as f:
        for k,v in parameters_dict.items():
            row = k + ": " + str(v) + "\n"
            f.write(row)



            
            
            


#--------------------------------#
#----------Parameters------------#
#--------------------------------#
root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/'
#input_file = root + 'processed_data/joined_data/all_months.pkl'

#Inputs
input_file = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/v15/matched_0.pkl'
version = 'v15' #v20

#Outputs
output_path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/processed_data/trained_models/'
output_cols = ['latitude_ERA', 'longitude_ERA','time','skt','MODIS_LST'] #we don't output all columns in our results

#Train/Test split
train_condition = pd.to_datetime("2019-01-01 00:00:00") #Everything less than this is used for training data
test_condition  = pd.to_datetime("2020-01-01 00:00:00") #Everything greater than this is used for test data. Everything not in these two groups is validation data by design
feature_names = ['sp', 'msl', 'u10', 'v10', 't2m',
            'aluvp', 'aluvd', 'alnip', 'alnid', 'istl1', 'istl2', 'sd', 'd2m','fal', 
            'skt', 
            'lsm',  'slt', 'sdfor','lsrh', 'cvh',  'z', 'isor', 'sdor', 'cvl','cl','anor', 'slor', 'sr', 'tvh', 'tvl']

target = ['MODIS_LST'] 


#Model parameters
epochs = 100
batch_size = 10000
use_validation_data = True #Do you want to use validation data for early stopping? Stopping conditions are defined in train_NN
optimizer = 'adam'

                 
            

#--------------------------------#
#--------------MAIN--------------#
#--------------------------------#

#Get the matched time variable data
df_variable = pd.read_pickle(input_file)


#Now get the constant data and join it on
constants_file= root +f'processed_data/ERA_timeconstant/ERA_constants_{version}.nc'
ERA_constants = xr.open_dataset(constants_file) 
df_constant = ERA_constants.to_dataframe().reset_index()
df_constant['longitude'] = (((df_constant.longitude + 180) % 360) - 180) #correct longitude

#This is the 'canonical' df we will be using 
df = pd.merge(df_variable, df_constant,  how='left', left_on=['latitude_ERA','longitude_ERA'], right_on = ['latitude','longitude'],suffixes=('', '_constant'))

    
#Normalise everything
features = df[feature_names]
target = df[target]
    
features_normed = (features-features.mean())/features.std()
target_normed = (target-target.mean())/target.std()


#Split data into training and testing set
split_data,results_df = train_test_split(df,train_condition,test_condition,
                                                            features_normed,target_normed,
                                                            output_cols)
print (len(split_data['x_train'].columns))
print (split_data['y_train'])
print (split_data['y_valid'])
#Train model
history,model = train_NN(split_data['x_train'],split_data['y_train'],
                         split_data['x_valid'],split_data['y_valid'],
                         epochs,batch_size,use_validation_data,optimizer)





#Make some predictions
predictions_normalized = model.predict(split_data['x_test'])
predictions = (predictions_normalized * target.std() ) + target.mean() #un-normalsie to get 'physical' value


#Bring together the test data and predictions into a single pandas df
results_df['predictions'] = predictions



#Save everything to disk
parameters_dict = {'input_file':     input_file,
                  'train_condition': train_condition,
                  'test_condition':  test_condition,
                  'epochs':          epochs,
                  'batch_size':      batch_size}
                                   


write_outputs(output_path+ID,model,history,results_df,parameters_dict)

    


print ("All completed OK")
