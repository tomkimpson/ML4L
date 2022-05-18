

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
import uuid

class NeuralNet():

    def __init__(self,cfg): 
        
        # Config file. Also get file itself to save to disk
        self.config = Config.from_json(cfg)
        self.train_json = cfg   

        # Data

        # Model Training parameters

        # IO

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
        self.node = None

        #Checks
        assert self.number_of_hidden_layers == len(self.nodes_per_layer)          # Number of layers = number of specified nodes



    def _load_data(self):

        """ Load the training and validation data"""
        self.training_data, self.validation_data = DataLoader().load_parquet_data(self.config.train)

    
        # CHECKS
        assert sorted(self.training_data.columns) == sorted(self.training_features + [self.target_variable])
        assert len(self.training_data.columns) - 1 == len(self.config.train.training_features) # Check number of columns is what we expect
        assert not self.training_data.isnull().any().any()   # Check for nulls
        assert not self.validation_data.isnull().any().any() # Check for nulls



    def _model_status(self):

        print ('Epochs:', self.epochs)
        print('Batch size:', self.batch_size)
        print('Number of features:',len(self.training_features))
        print ('Number of training samples:',len(self.training_data))
        print ('Learning rate:', self.LR)
        print ('Loss metric:', self.loss)
        print ('Number of hidden layers:', self.number_of_hidden_layers)
        print ('Nodes per layer:', self.node)
        print ('Selected features:', self.config.train.training_features)


        print ('Early stopping criteria:')
        print (self.early_stopping)

        print ('Checkpoint criteria')
        print (self.model_checkpoint)

       

    def _save_model(self):

        """Save model to disk after training """

        if os.path.exists(self.save_dir):
            print (self.save_dir, ' already exists')
            
            if self.overwrite is True:
                print ('Overwriting all contents')
                shutil.rmtree(self.save_dir)
                os.mkdir(self.save_dir)
            else:
                file_ID = str(uuid.uuid4().hex)
                fnew = self.save_dir+f'ML_{file_ID}/'
                self.save_dir = fnew
                print ('Not overwriting - creating a new directory called ',self.save_dir)
                os.mkdir(self.save_dir)

        else:
            os.mkdir(self.save_dir)
        
        print ('Saving model to:', self.save_dir)
        # Save the trained NN and the training history
        self.model.save(self.save_dir+'/trained_model') # TO SELF: How does this tie up with early stopping?  



        history_dict = self.history.history
        json.dump(history_dict, open(self.save_dir+'/training_history.json', 'w'))  # Save the training history
        json.dump(self.train_json, open(self. save_dir+'/configuration.json', 'w')) # Save the complete configuration used





    def _construct_network(self):

        """Construct the NN architecture and compile using Adam"""

        #Get the number of nodes for each layer. If none, defaults to nfeatures/2 for each layer
        if self.nodes_per_layer[0] is None:
            self.node = [int(self.nfeatures)/2]*self.nfeatures
        else:
            self.node = self.nodes_per_layer



        # Define network model
        self.model = tf.keras.Sequential(name='PredictLST')                   # Initiate sequential model
        self.model.add(tf.keras.layers.Dense(self.node[0],
                                             input_shape=(self.nfeatures,),
                                             activation="relu",
                                             name=f"layer_0"))                # Create first hidden layer with input shape
        for n in range(1,self.number_of_hidden_layers):                       # Iterate over remaining hidden layers
            self.model.add(tf.keras.layers.Dense(
                                            self.node[n],
                                            activation="relu",
                                            name=f"layer_{n}"))

        self.model.add(tf.keras.layers.Dense(1, name='output'))                # And finally define an output layer 


        #Compile it
        opt = tf.keras.optimizers.Adam(learning_rate=self.LR) #Always use Adam
        self.model.compile(optimizer=opt,
                           loss=self.loss,
                           metrics=self.metrics)


    def _callbacks(self):

        """Define early stopping and checkpoint callbacks"""


        self.early_stopping = EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=self.stopping_patience,
                                            verbose=1,
                                            mode='auto',
                                            baseline=None,
                                            restore_best_weights=True)



        self.model_checkpoint = ModelCheckpoint(filepath = self.path_to_trained_models+'tmp_checkpoint', 
                                                monitor='val_loss', 
                                                save_best_only=True, 
                                                mode='min',
                                                save_freq='epoch',
                                                period=self.epoch_save_freq #period argument is deprecated, but works well 
                                                )





    def _train_network(self):

        """Train the model"""
   
        print('Training network with the following parameters:')
        self._model_status()

        #Train it 
        self.history = self.model.fit(self.training_data[self.training_features], 
                                      self.training_data[self.target_variable], 
                                      epochs=self.epochs, batch_size=self.batch_size,
                                      verbose=1,
                                      validation_data=(self.validation_data[self.training_features], self.validation_data[self.target_variable]),
                                      callbacks=[self.early_stopping,self.model_checkpoint]
                                     ) 

        print(self.model.summary())
       




    def train(self):

        self._load_data()

        self._construct_network()

        self._callbacks()
        
        self._train_network
        
        self._save_model()  

        #Drop large files explicitly
        del self.training_data
        del self.validation_data




    def _predict_status(self):

        print ('Making predictions with the following settings:')
        print ('Trained model:', self.save_dir)
        print ('Features:', self.training_features)





    def predict(self):
        
        #Load
        loaded_model = tf.keras.models.load_model(self.save_dir+'/trained_model') # Load the model
        cols = self.training_features + [self.target_variable]
        test_data = pd.read_parquet(self.config.train.testing_data,columns=cols) # Load the test data

        self._predict_status()
        
        #Predict
        predictions = loaded_model.predict(test_data[self.training_features])
        print ('Predictions completed')  
        del test_data

        #IO
        meta_data = pd.read_parquet(self.config.train.testing_data,columns=['latitude_ERA', 'longitude_ERA','time','MODIS_LST'])
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
    