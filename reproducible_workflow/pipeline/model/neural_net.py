

#internal
from .base_model import BaseModel 
from dataloader.dataloader import DataLoader

#external
import tensorflow as tf
import numpy as np



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
        
    def load_data(self):
        self.training_data, self.validation_data = DataLoader().load_data(self.config.data)
        self.training_data = self.training_data.map(self._parse_function)
        self.training_data = self.validation_data.map(self._parse_function)
        self._preprocess_data()
            
    def _preprocess_data(self):
        """Reads TFRecords """
        
        self.training_data = self.training_data.shuffle(2048)
        self.training_data = self.training_data.prefetch(buffer_size=AUTOTUNE)
        self.training_data = self.training_data.batch(1024)
        pass
        
        
    def _parse_function(self,example_proto):
    
        feature_description = {
        'feature_input': tf.io.FixedLenFeature([51,], tf.float32,default_value=np.zeros(51,)),
        'label' : tf.io.FixedLenFeature([1, ], tf.float32,default_value=np.zeros(1,))
        }

        example = tf.io.parse_single_example(example_proto, feature_description)

        image = example['feature_input']
        label = example['label']



        return image,label    
    



    #----------------DEV AREA
    def load_data_alternative(self):
        self.training_data, self.validation_data = DataLoader().load_data(self.config.data)
        #self.training_data = self.training_data.map(self._parse_function)
        #self.training_data = self.validation_data.map(self._parse_function)
        #self._preprocess_data()


    def construct_network(self):

        model = tf.keras.Sequential()
        print(self.training_features)
        for n in range(self.number_of_hidden_layers):
            model.add(tf.keras.layers.Dense(2,input_shape=(5,1),activation="relu",name=f"layer_{n}"))

        model.add(tf.keras.layers.Dense(1, name='output'))


        print ('created a model')
        print(model.summary())
