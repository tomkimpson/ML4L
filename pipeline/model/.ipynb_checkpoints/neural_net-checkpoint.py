

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
        self.training_data = None
        self.validation_data = None   
        
        
        
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
    
    # #instance method: functions defined inside a class that can only be called from an instance of that class
    # def description(self):
    #     return f"{self.name} is {self.age} years old"

    # def speak(self):
    #     return f"{self.name} says {sound}"

    # def __str__(self):
    #     return f"{self.name} is a very good boy who is {self.age} years old"

    # def __repr__(self):
    #     return f"{self.name} is a very good boy in the repr who is {self.age} years old"