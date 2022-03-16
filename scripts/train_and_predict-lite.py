import tensorflow as tf
import os
import time
import json
import pandas as pd



"""
Script to train a sequential NN.
NN trains on training data, all results output to disk
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
    
    
    
    

    
input_file = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/MODIS_ERA_joined_data_averaged.pkl'
train_condition = pd.to_datetime("2019-01-01 00:00:00")
test_condition  = pd.to_datetime("2020-01-01 00:00:00")
epochs = 100
batch_size = 100000
output_path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/'

     
    
def load_and_process_data(input_file,train_condition,test_condition):
    
    print('Loading the data')
    
    #Load the raw
    df = pd.read_pickle(input_file)
    
    
    
    # Create a "results_df" copy that will be used later for clean IO
    results_df = df[['latitude_ERA', 'longitude_ERA','time','t2m','MODIS_LST']].copy()
    

    # Split into features/targets 
    feature_names = [ 'sp', 'msl', 'u10', 'v10','t2m',
                         'aluvp', 'aluvd', 'alnip', 'alnid', 'cl',
                         'cvl', 'cvh', 'slt', 'sdfor', 'z', 'sd', 'sdor', 'isor', 'anor', 'slor',
                         'd2m', 'lsm', 'fal'] 


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
    history = model.fit(x, y, epochs=epochs, batch_size=batch_size) 
    

    return history,model
    



def write_outputs(output_path,model,history,df,parameters_dict):

    
    
    
    #Create a directory
    id = int(time.time())
    fout = output_path+f'ML_{str(id)}/'
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
        f.write(json.dumps(parameters_dict)) # use `json.loads` to do the reverse
    
    

    
#Get data
x_train,y_train,x_test,y_test,T_normalize_mean,T_normalize_std,results_df = load_and_process_data(input_file,train_condition,test_condition)

#Train model
history,model = train_NN(x_train,y_train,epochs,batch_size)

#Make some predictions
predictions_normalized = model.predict(x_test)
predictions = (predictions_normalized * T_normalize_std ) + T_normalize_mean

#Bring together the test data and predictions into a single pandas df
results_df['predictions'] = predictions


#Save everything to disk
parameters_dict = {'input_file':     input_file,
                  'train_condition': train_condition,
                  'test_condition':  train_condition,
                  'epochs':          epochs,
                  'batch_size':      batch_size}
                                   





write_outputs(output_path,model,history,results_df,parameters_dict)

    


print ("All completed OK")