




import tensorflow as tf
import os
import time
import json
import pandas as pd



"""
Script to train a sequential NN.
Takes a df, filters based on `condition` (default None), and separates into test/train based on time
NN trains on training data, all results output to disk
"""


def train_test_split(df,filter_condition,train_condition, test_condition,features,targets):
    
    
    """
    Separate df into a train and test set.
    Returns training and testing dfs as well as split into inputs/outputs 
    """
    
    #Filter dataset
    if filter_condition is not None:
        df_filtered = df.query(filter_condition)
    else:
        df_filtered = df
    
    #Select train/test data
    training_data = df_filtered.query(train_condition)
    test_data     = df_filtered.query(test_condition)
    
    
    #Separate into features/targets

    x_train = training_data[features]
    y_train = training_data[targets]

    x_test = test_data[features]
    y_test = test_data[targets]
    
    
    return x_train,y_train,x_test,y_test,training_data, test_data #
    
    


def create_normalizer_layer(x_train):
    #Create a normaliser layer
    print ('Creating a normalization layer')
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_train)
    
    return normalizer

def train_NN(x_train,y_train,normalizer):


    #Check GPU available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    #Create a basic NN model
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(int(len(features)/2), activation='relu',input_shape=(len(features),),name='layer1'),
        tf.keras.layers.Dense(1, name='output')
  ])

    #Compile it
    print ('Compiling model')
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    

    
    #Train it
    print('Training model')
    history = model.fit(x_train, y_train, epochs=100, batch_size=10000) 
    
    
    return history, model


def write_outputs(output_path,model,history,x_train,y_train,x_test,y_test,df_train,df_test):

    print ('Writing outputs to dir: ', fout)
    id = int(time.time())
    fout = output_path+f'ML_{str(id)}/'
    os.makedirs(fout)
    print ('Writing outputs to dir: ', fout)


    
    #NN
    model.save(fout+'trained_model') 
    history_dict = history.history
    json.dump(history_dict, open(fout+'history.json', 'w'))
    
    #Data
    #Probaby overkill saving all of these
    x_train.to_pickle(fout + "x_train.pkl") 
    y_train.to_pickle(fout + "y_train.pkl") 
    x_test.to_pickle(fout + "x_test.pkl") 
    x_test.to_pickle(fout + "y_test.pkl")
    df_train.to_pickle(fout + "df_train.pkl") 
    df_test.to_pickle(fout + "df_test.pkl") 



    
    


def pipeline(input_file,output_path,filter_condition,train_condition, test_condition,features):
    
    
    #Load the data
    print('Loading the data')
    df = pd.read_pickle(input_file)

    
    #Process into train/test
    targets = ['MODIS_LST']
    x_train,y_train,x_test,y_test,df_train,df_test = train_test_split(df,filter_condition,train_condition, test_condition,features,targets)

    
    
    #Train NN
    normalizer = create_normalizer_layer(x_train)
    history,model = train_NN(x_train,y_train,normalizer)
    
    
    #Save trained NN in new dir, along with train/test sets
    write_outputs(output_path,model,history,x_train,y_train,x_test,y_test,df_train,df_test)

    
    
    
#Parameters

#IO
input_file = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/MODIS_ERA_joined_data_averaged.pkl'
output_path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/'



#Pre Processing
filter_condition = None
train_condition = 'time < "2019-01-01 00:00:00"'
test_condition = 'time >= "2020-01-01 00:00:00"'
features = ['sp', 'msl', 'u10', 'v10','t2m',
            'aluvp', 'aluvd', 'alnip', 'alnid', 'cl',
            'cvl', 'cvh', 'slt', 'sdfor', 'z', 'sd', 'sdor', 'isor', 'anor', 'slor',
            'd2m', 'lsm', 'fal'] 




#Go
pipeline(input_file,output_path,filter_condition,train_condition, test_condition,features)

