import tensorflow as tf
import os
import time
import json
import pandas as pd
import numpy as np


"""
Script to train a sequential NN.
NN trains on training data, all results output to disk.
Use this script over `train_and_predict.py`
"""

#GPU configuration - dont use too much memory
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
    
    
    
    

    
#input_file = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/MODIS_ERA_joined_data_single.pkl'
input_file = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/ML_data_ERA_MODIS_joined.pkl'



train_condition = pd.to_datetime("2019-01-01 00:00:00")
test_condition  = pd.to_datetime("2020-01-01 00:00:00")
epochs = 100
batch_size = 10000
output_path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/'

        
#all_features = [ 'sp', 'msl', 'u10', 'v10','t2m',
 #                        'aluvp', 'aluvd', 'alnip', 'alnid', 'cl',
  #                       'cvl', 'cvh', 'slt', 'sdfor', 'z', 'sd', 'sdor', 'isor', 'anor', 'slor',
   #                      'd2m', 'lsm', 'fal','skt'] 
    

all_features = [ 'isor', 'anor', 'slor','d2m', 'lsm', 'fal','skt'] 

parameters_dict = {'input_file':     input_file,
                  'train_condition': train_condition,
                  'test_condition':  test_condition,
                  'epochs':          epochs,
                  'batch_size':      batch_size}


    
def train_test_split(df,train_condition,test_condition,normalised_features,normalised_targets):
    
   
    
    # Create a "results_df" copy that will be used later for clean IO
    results_df = df[['latitude_ERA', 'longitude_ERA','time','skt','MODIS_LST']].copy()
      
    
    #Separate into train/test based on time
    idx_train = df.time < train_condition 
    idx_test = df.time > test_condition 

    x_train = normalised_features[idx_train] 
    y_train = normalised_targets[idx_train]

    x_test = normalised_features[idx_test]
    y_test = normalised_targets[idx_test]
    results_df = results_df[idx_test] #For IO we will just output predictions
    
    
    return x_train,y_train,x_test,y_test,target.mean(),target.std(),results_df
    
    
    

def train_NN(x,y,epochs,batch_size):
    
    
    print('Training model')
    nfeatures = x.shape[-1]


    #Create a basic NN model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(int(nfeatures/2), activation='relu',input_shape=(nfeatures,),name='layer1'),
        tf.keras.layers.Dense(1, name='output')
      ])

    #Compile it
    
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    

    
    #Train it
    history = model.fit(x, y, epochs=epochs, batch_size=batch_size,verbose=1) 
    

    return history,model
    


    
    
def write_outputs(output_path,model,history,df,parameters_dict,identifier):

    
    
    
    #Create a directory
    fout = output_path+f'ML_FI_{str(identifier)}/'
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

        
    
    
    
def pipeline(df,train_condition,test_condition,selected_normalised_features,target_normed,popped_feature):
    
    print('Splitting data')
    x_train,y_train,x_test,y_test,T_normalize_mean,T_normalize_std,results_df = train_test_split(df,train_condition,test_condition,selected_normalised_features,target_normed)
    
    #Train model
    print ('Training model')
    history,model = train_NN(x_train,y_train,epochs,batch_size)

    
    
    
    #Predict 
    #Make some predictions
    print('Predict')
    predictions_normalized = model.predict(x_test)
    predictions = (predictions_normalized * T_normalize_std ) + T_normalize_mean
    
    
    #Bring together the test data and predictions into a single pandas df
    results_df['predictions'] = predictions
    
    
    
    
    #IO
    write_outputs(output_path,model,history,results_df,parameters_dict,popped_feature)


    
    
  

    #Evaluate
    #print('Evaluating model')
    #score = model.evaluate(x_test, y_test,batch_size)

    #Make as a pandas df
    # d= {'feature': [popped_feature], 'score': [score]}
    # dftmp = pd.DataFrame(data=d)
    
   # return dftmp 
    
    

    
print('Loading the data')
#Load the raw
df = pd.read_pickle(input_file)

#Normalise everything
features = df[all_features]
target = df['MODIS_LST']
    
#Normalise
target_normed = (target-target.mean())/target.std()
features_normed = (features-features.mean())/features.std()


#Null hypothesis, using all features
#pipeline(df,train_condition,test_condition,features_normed,target_normed,'H0')
    

#Iterate
#dfs = []
for i in all_features:
    print('Pop feature:', i)
    
    #Use all features apart from one
    feature_names = [n for n in all_features if n != i]
    selected_normalised_features = features_normed[feature_names]
    
    pipeline(df,train_condition,test_condition,features_normed,target_normed,i)
    
    
    #dfs.append(dftmp)   


    
    
#Add the unpermuted score
# dfs.append(df_null)
# print('IO')
# #Bring together and save to disk
# df = pd.concat(dfs).reset_index()
# df.to_pickle('small_data/training_feature_importance.pkl')







print ("All completed OK")
