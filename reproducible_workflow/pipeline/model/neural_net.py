

#internal
from .base_model import BaseModel 
from dataloader.dataloader import DataLoader

#external
import tensorflow as tf
import numpy as np
import os


class NeuralNet(BaseModel):

    #species = "Canis familiaris" #this is a class attribute. True for ALL dogs

    def __init__(self,config): 
        super().__init__(config) #call the constructor, aka the init of the parent class
        
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


        #Checks
        assert not os.path.exists(self.path_to_trained_models + self.model_name)  # Save directory does not already exist
        assert self.number_of_hidden_layers == len(self.nodes_per_layer)          # Number of layers = number of specified nodes


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
    



    #----------------DEV AREA
    def load_data_alternative(self):
        self.training_data, self.validation_data = DataLoader().load_data(self.config.data)
        #self.training_data = self.training_data.map(self._parse_function)
        #self.training_data = self.validation_data.map(self._parse_function)
        #self._preprocess_data()


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


    def train_network(self):

        print('Training network with:')
        self._model_status()

        history = self.model.fit(self.training_data[self.training_features], 
                                 self.training_data[self.target_variable], 
                                 epochs=self.epochs, batch_size=self.batch_size,
                                 verbose=1,
                                 validation_data=(self.validation_data[self.training_features], self.validation_data[self.target_variable]),
                                 ) 
        return history



    def _model_status(self):

        print ('Epochs:', self.epochs)
        print('Batch size:', self.batch_size)
        print('Number of features:',len(self.training_features))
        print ('Number of training samples:',len(self.training_data))
        print(self.model.summary())


    def save_model(self):

        """Save model to disk after training """
        
        #Create a directory where we will save everything
        os.mkdir(self.path_to_trained_models + self.model_name)

    #  "train": {
    #     "batch_size": 10000,
    #     "buffer_size": 1000,
    #     "epochs": 2,
    #     "number_of_hidden_layers":2,
    #     "nodes_per_layer": [None,None],
    #     "val_subsplits": 5,
    #     "training_features": ['sp', 'msl', 'u10', 'v10'],
    #     "target_variable": 'MODIS_LST',
    #     "learning_rate": 3e-4,
    #     "loss": 'mse',
    #     "metrics": ["accuracy"],
    #     "path_to_trained_models": f'{root}processed_data/trained_models/',
        
        # #Create a directory
        # os.makedirs(fout)
        # print ('Writing data to:', fout)

        
        # #Save the trained NN and the training history
        # model.save(fout+'trained_model') 
        # history_dict = history.history
        # json.dump(history_dict, open(fout+'history.json', 'w'))
    
        
        # #Save meta data as a txt file
        # with open(fout+'meta.txt', 'w') as f:
        #     for k,v in parameters_dict.items():
        #         row = k + ": " + str(v) + "\n"
        #         f.write(row)

        # return fout

