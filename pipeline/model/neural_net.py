

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
from collections import Counter
class NeuralNet():

    def __init__(self,cfg): 
        
        # Config file. Also get file itself to save to disk
        self.config = Config.from_json(cfg)
        self.train_json = cfg   

        # Data
        self.training_features = self.config.train.training_features
        self.target_variable   = self.config.train.target_variable
        #self.nfeatures         = len(self.training_features)

        # Model Training parameters
        self.batch_size               = self.config.train.batch_size
        self.epochs                   = self.config.train.epochs
        self.number_of_hidden_layers  = self.config.train.number_of_hidden_layers
        self.nodes_per_layer          = self.config.train.nodes_per_layer
        self.LR                       = self.config.train.learning_rate
        self.metrics                  = self.config.train.metrics
        self.loss                     = self.config.train.loss
        self.pretrained_model         = self.config.train.pretrained_model


        #Feature importance and Permutation
        self.features_to_permute      = self.config.permute.features_to_permute
        
        # IO
        self.epoch_save_freq        = self.config.train.epoch_save_freq
        self.stopping_patience      = self.config.train.early_stopping_patience
        self.path_to_trained_models = self.config.train.path_to_trained_models
        self.model_name             = self.config.train.model_name
        self.save_dir               = self.path_to_trained_models + self.model_name
        self.overwrite              = self.config.train.overwrite

        #Assignments
        self.training_data   = None
        self.validation_data = None   
        self.test_data       = None
        self.model           = None
        self.node            = None
        self.history         = None
        self.selected_training_features = None

    def _load_data(self,kind):

        """ Load either the training and validation data OR the test data"""

        if kind == 'train':
            self.training_data, self.validation_data = DataLoader().load_training_data(self.config.train)
            # CHECKS
            assert sorted(self.training_data.columns) == sorted(self.training_features + [self.target_variable]) #Check columns in df are the ones we expect
            assert len(self.training_data.columns) - 1 == len(self.training_features) # Check number of columns is what we expect
            assert not self.training_data.isnull().any().any()   # Check for nulls
            assert not self.validation_data.isnull().any().any() # Check for nulls
        
        if kind == 'test':
            self.json_read_cols, self.test_data = DataLoader().load_testing_data(self.config)

    def _model_status(self):

        print ('-------------------------------------------------------------')

        print ('Epochs:', self.epochs)
        print('Batch size:', self.batch_size)
        print('Number of features:',len(self.training_features))
        print ('Number of training samples:',len(self.training_data))
        print ('Learning rate:', self.LR)
        print ('Loss metric:', self.loss)
        print ('Number of hidden layers:', self.number_of_hidden_layers)
        print ('Nodes per layer:', self.node)
        print ('-------------------------------------------------------------')
        print ('Selected features:')
        for i in range(0, len(self.selected_training_features), 5):
            print(*self.selected_training_features[i:i+5], sep=' ')
        print ('-------------------------------------------------------------')


        print ('Early stopping criteria:')
        for k, v in vars(self.early_stopping).items():
            print(f'{k : <30} {v}')
        print ('-------------------------------------------------------------')


        print ('Checkpoint criteria')
        for k, v in vars(self.model_checkpoint).items():
            print(f'{k : <30} {v}')
        print ('-------------------------------------------------------------')


        print ('-------------------------------------------------------------')
        print ('-------------------------------------------------------------')

    def _predict_status(self,cols):
        print ('-------------------------------------------------------------')
        print ('Making predictions with the following settings:')
        print ('-------------------------------------------------------------')
        print ('Trained model:', self.save_dir)
        print ('Features:', cols)
        print ('-------------------------------------------------------------')

    def _create_directory(self):
 
        """Create a directory to save trained model to"""
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
            print ('Creating a new directory called:',self.save_dir)
            os.mkdir(self.save_dir)

    def _save_model(self):
        """Save model to disk after training"""
        
        print ('Saving model to:', self.save_dir)
        # Save the trained NN and the training history
        self.model.save(self.save_dir+'/trained_model')  



        history_dict = self.history.history
        json.dump(history_dict, open(self.save_dir+'/training_history.json', 'w'))  # Save the training history
        json.dump(self.train_json, open(self. save_dir+'/configuration.json', 'w')) # Save the complete configuration used

    def _construct_network(self,permuted_feature):

        """Construct the NN architecture and compile using Adam.
           Drops permuted_feature from the list of training features to
           allow for drop column feature importance tests"""

        #Drop this feature if it exists
        if permuted_feature is not None:
            self.selected_training_features = self.training_features
            self.selected_training_features.remove(permuted_feature)
        else:
            self.selected_training_features = self.training_features
        self.n_selected_features = len(self.selected_training_features)


        #Get the number of nodes for each layer. If none, defaults to nfeatures/2 for each layer
        if self.nodes_per_layer[0] is None:
            self.node = [int(self.n_selected_features/2)]*self.number_of_hidden_layers
        else:
            self.node = self.nodes_per_layer

        assert self.number_of_hidden_layers == len(self.node)          # Number of layers = number of specified nodes


        # Define network model
        self.model = tf.keras.Sequential(name='PredictLST')                   # Initiate sequential model
        self.model.add(tf.keras.layers.Dense(self.node[0],
                                             input_shape=(self.n_selected_features,),
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



        self.model_checkpoint = ModelCheckpoint(filepath = self.save_dir+'/checkpoint', 
                                                monitor='val_loss', 
                                                save_best_only=True, 
                                                mode='min',
                                                save_freq='epoch',
                                                period=self.epoch_save_freq #period argument is deprecated, but works well 
                                                )

    def _train_network(self):

        """Train the model"""
        print ('-------------------------------------------------------------')
        print('Training network with the following parameters:')
        self._model_status()
 
        
        #Train it 
        self.history = self.model.fit(self.training_data[self.selected_training_features], 
                                      self.training_data[self.target_variable], 
                                      epochs=self.epochs, batch_size=self.batch_size,
                                      verbose=1,
                                      validation_data=(self.validation_data[self.selected_training_features], self.validation_data[self.target_variable]),
                                      callbacks=[self.early_stopping,self.model_checkpoint]
                                     ) 

        print(self.model.summary())
       
    def _evaluate_model(self):

        """Evaluate the model"""
        print ('-------------------------------------------------------------')
        print(f'Evaluating the trained model')
        print ('Eval test data is:')
        print(self.test_data[self.selected_training_features])
        print ('Eval target is')
        print(self.test_data[self.target_variable])
        print ('Eval batchsize is')
        print(self.batch_size)
        print ('Defined globals = ')
        print (globals())
        score = self.model.evaluate(self.test_data[self.selected_training_features], 
                                    self.test_data[self.target_variable],
                                    batch_size=self.batch_size)



        #Drop large files explicitly
        print ('Dropping test data')
        del self.test_data

        return score       


    def train(self):

        self._load_data(kind='train')

        if self.pretrained_model is None:
            self._create_directory()
            self._construct_network(None)
        else:
             print ('Loading a pretrained model from ', self.pretrained_model)
             self.model = tf.keras.models.load_model(self.pretrained_model)
        
        self._callbacks()
        self._train_network()
        self._save_model()  

        #Drop large files explicitly
        print ('Dropping train/validate data')
        del self.training_data
        del self.validation_data

    def predict(self):
        
        #Load
        loaded_model = tf.keras.models.load_model(self.save_dir+'/trained_model') # Load the model

        with open(self.save_dir+'/configuration.json') as f:
            config=json.load(f)
            cols = config['train']['training_features']     #Read from the config file saved with the model which features were used for training and use these same features when testing
        
        test_data = pd.read_parquet(self.config.predict.testing_data,columns=cols) # Load the test data
        print(test_data)
        self._predict_status(cols)
        
        #Predict
        predictions = loaded_model.predict(test_data[cols])
        print ('Predictions completed')  
        del test_data 

        #IO
        meta_data = pd.read_parquet(self.config.predict.testing_data,columns=['latitude_ERA', 'longitude_ERA','time','MODIS_LST','skt_unnormalised'])
        meta_data['predictions'] = predictions 
        fout = self.save_dir + '/predictions.parquet'
        print ('Saving to:',fout)
        meta_data.to_parquet(fout,compression=None)

    def evaluate(self):

        """Evaluate the trained model on the test data and also perform a drop column feature importance test"""

        print('-------------------EVALUATING-------------------')


        #Load the test data
        self._load_data(kind='test')
        # Load the pre-trained model
        self.model = tf.keras.models.load_model(self.save_dir+'/trained_model') # Load the model
        
        #Evaluate the model with all its features and save this score to array
        model_score = self._evaluate_model()
        all_features = ['Model']
        all_scores = [model_score]
        
        #Iterate over permuted features

        print ('ITERATING OVER', self.features_to_permute)
        for feature in  self.features_to_permute:
            print('Permuting feature:', feature)

            self._load_data(kind='train')
            self._construct_network(feature)
            self._callbacks()
            self._train_network() #train and validate data dropped here

            self._load_data(kind='test')
            score = self._evaluate_model()
            print (self.model.metrics_names)
            
            print ('Feature/Score',feature,score)
            
            #Output to arrays
            all_features.append(feature)   
            all_scores.append(score)         
      
    

        #IO
        df_scores = pd.DataFrame(data = {'features': all_features, 'scores': all_scores})
        fout = self.save_dir + '/scores.parquet'
        print ('Saving to:',fout)
        df_scores.to_parquet(fout,compression=None)



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
    