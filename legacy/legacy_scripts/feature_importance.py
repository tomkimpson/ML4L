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
epochs = 10
batch_size = 10000
output_path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/'

n_permutations = 5
    
def load_and_process_data(input_file,train_condition,test_condition):
    
    print('Loading the data')
    
    #Load the raw
    df = pd.read_pickle(input_file)
    
    
    
    # Create a "results_df" copy that will be used later for clean IO
    results_df = df[['latitude_ERA', 'longitude_ERA','time','skt','t2m','MODIS_LST']].copy()
    

    # Split into features/targets 
    feature_names = [ 'sp', 'msl', 'u10', 'v10','t2m',
                         'aluvp', 'aluvd', 'alnip', 'alnid', 'cl',
                         'cvl', 'cvh', 'slt', 'sdfor', 'z', 'sd', 'sdor', 'isor', 'anor', 'slor',
                         'd2m', 'lsm', 'fal','skt'] 


    features = df[feature_names]
    target = df.pop('MODIS_LST')
    
    #Normalise
    target_normed = (target-target.mean())/target.std()
    features_normed = (features-features.mean())/features.std()
    
    
    #Separate into train/test based on time
    idx_train = df.time < train_condition 
    idx_test = df.time > test_condition 

    x_train = features_normed[idx_train] 
    y_train = target_normed[idx_train]

    x_test = features_normed[idx_test]
    y_test = target_normed[idx_test]
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
    


            
#Load and normalise data
print('Getting data')
x_train,y_train,x_test,y_test,T_normalize_mean,T_normalize_std,results_df = load_and_process_data(input_file,train_condition,test_condition)

#Train model
print ('Training model')
history,model = train_NN(x_train,y_train,epochs,batch_size)


#Evaluate
print('Evaluating model')
score0 = model.evaluate(x_test, y_test,batch_size)


#Permute
print('Feature importance')
dfs = []

for i in x_test.columns:

    running_score = 0
    for j in range(n_permutations):
    
        #Create a copy that will be permuted
        x_test_permuted = x_test.copy()
    
        #Shuffle target column
        shuffle = np.random.permutation(x_test_permuted[i])
    
        #Permute
        x_test_permuted[i] = shuffle
    
        #Now make a prediction with the permuted data
        score = model.evaluate(x_test_permuted, y_test,batch_size)
        
        running_score += score 
        
        
    running_score = running_score/n_permutations
    #Make as a pandas df
    d= {'feature': [i], 'score': [running_score]}
    dftmp = pd.DataFrame(data=d)
    dfs.append(dftmp)    
        

#Add the unpermuted score
d= {'feature': ['H0'], 'score': [score0]}
dftmp = pd.DataFrame(data=d)
dfs.append(dftmp)

print('IO')
#Bring together and save to disk
df = pd.concat(dfs).reset_index()
df.to_pickle('feature_importance_n5.pkl')







print ("All completed OK")
