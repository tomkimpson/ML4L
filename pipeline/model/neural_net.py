

#internal
#from .base_model import BaseModel 
from utils.config import Config 
from dataloader.dataloader import DataLoader

#external
import tensorflow as tf
import numpy as np
import os
import json
import shutil
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import pandas as pd

class NeuralNet():

    def __init__(self,cfg): 
        #super().__init__(config) #call the constructor, aka the init of the parent class
        

        self.config = Config.from_json(cfg)
        self.train_json = cfg   

        self.batch_size = self.config.train.batch_size
        self.epochs = self.config.train.epochs
        self.number_of_hidden_layers = self.config.train.number_of_hidden_layers
        self.nodes_per_layer = self.config.train.nodes_per_layer
        self.training_data = None
        self.validation_data = None   
        
        self.training_features = self.config.train.training_features
        self.LR = self.config.train.learning_rate
        self.metrics = self.config.train.metrics
        self.loss = self.config.train.loss
        self.model = None
        self.target_variable = self.config.train.target_variable
        self.training_features = self.config.train.training_features
        self.nfeatures = len(self.training_features)
        self.path_to_trained_models = self.config.train.path_to_trained_models
        self.model_name = self.config.train.model_name
        self.overwrite = self.config.train.overwrite
        self.epoch_save_freq = self.config.train.epoch_save_freq
        self.stopping_patience = self.config.train.early_stopping_patience
        self.save_dir = self.path_to_trained_models + self.model_name


        #Checks
        if os.path.exists(self.path_to_trained_models + self.model_name):
            print (self.path_to_trained_models + self.model_name, ' exists')
            # if self.overwrite is False: 
            #     raise Exception( f'Save directory {self.path_to_trained_models + self.model_name} already exists')
            # if self.overwrite is True: 
            #     print ('Overwriting model directory: ', self.model_name)
            #     shutil.rmtree(self.path_to_trained_models + self.model_name)


        assert self.number_of_hidden_layers == len(self.nodes_per_layer)          # Number of layers = number of specified nodes



    def load_data(self):
        self.training_data, self.validation_data = DataLoader().load_parquet_data(self.config.train)
        
        assert len(self.training_data.columns) - 1 == len(self.config.train.training_features) #Check number of columns is what we expect


    def construct_network(self):

        #Get the number of nodes for each layer
        #If none, defaults to nfeatures/2 for each layer
        if self.nodes_per_layer[0] is None:
            node = [int(self.nfeatures)/2]*self.nfeatures
        else:
            node = self.nodes_per_layer


        # Define network model
        self.model = tf.keras.Sequential(name='PredictLST')                               # Initiate sequential model
        self.model.add(tf.keras.layers.Dense(node[0],
                                        input_shape=(self.nfeatures,),
                                        activation="relu",
                                        name=f"layer_0"))                # Create first hidden layer with input shape
        for n in range(1,self.number_of_hidden_layers):                  # Iterate over remaining hidden layers
            self.model.add(tf.keras.layers.Dense(node[n],activation="relu",name=f"layer_{n}"))

        self.model.add(tf.keras.layers.Dense(1, name='output'))          # And finally define an output layer 


        #Compile it
        opt = tf.keras.optimizers.Adam(learning_rate=self.LR) #Always use Adam
        self.model.compile(optimizer=opt,
                           loss=self.loss,
                           metrics=self.metrics)


        #Define early stopping criteria
        self.early_stopping = EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=self.stopping_patience,
                                            verbose=1,
                                            mode='auto',
                                            baseline=None,
                                            restore_best_weights=True)
    
        #Checkpoints
        self.model_checkpoint = ModelCheckpoint(filepath = self.path_to_trained_models+'tmp_checkpoint', 
                                                monitor='val_loss', 
                                                save_best_only=True, 
                                                mode='min',
                                                save_freq=int(self.epoch_save_freq * len(self.training_data) / self.batch_size), #save best model every epoch_save_freq epochs
                                                )

#c#heckpoint= keras.callbacks.ModelCheckpoint(filepath= checkpoint_filepath, verbose=1, save_freq="epoch", mode='auto',monitor='val_loss', save_best_only=True)


    def train_network(self):

        print('Training network with:')
        self._model_status()

        self.history = self.model.fit(self.training_data[self.training_features], 
                                      self.training_data[self.target_variable], 
                                      epochs=self.epochs, batch_size=self.batch_size,
                                      verbose=1,
                                      validation_data=(self.validation_data[self.training_features], self.validation_data[self.target_variable]),
                                      callbacks=[self.early_stopping,self.model_checkpoint]
                                     ) 

        for key in self.history.history:
            print(key)
        

    def _model_status(self):

        print ('Epochs:', self.epochs)
        print('Batch size:', self.batch_size)
        print('Number of features:',len(self.training_features))
        print ('Number of training samples:',len(self.training_data))
        print(self.model.summary())


    def save_model(self):

        """Save model to disk after training """

        #Create a directory where we will save everything
        os.mkdir(save_dir)
        print ('Saving model to:', save_dir)
        # Save the trained NN and the training history
        self.model.save(save_dir+'/trained_model') 
        history_dict = self.history.history
        json.dump(history_dict, open(save_dir+'/training_history.json', 'w'))          # Save the training history
        json.dump(self.train_json, open(save_dir+'/configuration.json', 'w')) # Save the complete configuration used



    def predict(self):
        print('loading model at:', self.save_dir+'/trained_model')
        print(self.config.train.testing_data)
        loaded_model = tf.keras.models.load_model(self.save_dir+'/trained_model') # Load the model
        test_data = pd.read_parquet(self.config.train.testing_data,columns=self.config.train.training_features + [self.config.train.target_variable]) #Load the test data
        print ('Got the test data and model')

        print(f'Using trained model {self.save_dir} to make some predictions')
        predictions = loaded_model.predict(test_data[self.training_features])                 # Prediction 

        #Drop test data
        del test_data

        #Load just the meta info
        meta_data = pd.read_parquet(self.config.train.testing_data,columns=['latitude_ERA', 'longitude_ERA','time','skt','MODIS_LST'])
        print ('len loaded df:', len(meta_data))
        print('len preds', len(predictions))
        meta_data['predictions'] = predictions 
        
        fout = self.save_dir + '/predictions.parquet'
        print ('Saving to:',fout)
        meta_data.to_parquet(fout,compression=None)



#Make some predictions










#---------------SCRATCH AREA---------------------#



    # def load_data(self):
    #     self.training_data, self.validation_data = DataLoader().load_data(self.config.data)
    #     self.training_data = self.training_data.map(self._parse_function)
    #     self.training_data = self.validation_data.map(self._parse_function)
    #     self._preprocess_data()
            
    # def _preprocess_data(self):
    #     """Reads TFRecords """
        
    #     self.training_data = self.training_data.shuffle(2048)
    #     self.training_data = self.training_data.prefetch(buffer_size=AUTOTUNE)
    #     self.training_data = self.training_data.batch(1024)
    #     pass
        
        
    # def _parse_function(self,example_proto):
    
    #     feature_description = {
    #     'feature_input': tf.io.FixedLenFeature([51,], tf.float32,default_value=np.zeros(51,)),
    #     'label' : tf.io.FixedLenFeature([1, ], tf.float32,default_value=np.zeros(1,))
    #     }

    #     example = tf.io.parse_single_example(example_proto, feature_description)

    #     image = example['feature_input']
    #     label = example['label']



    #     return image,label    
    