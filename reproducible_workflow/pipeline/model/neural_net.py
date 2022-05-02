

#internal
from .base_model import BaseModel 





class Dog(BaseModel):

    species = "Canis familiaris" #this is a class attribute. True for ALL dogs

    def __init__(self,config,name,age): #this defines the properties that all Dog objects must have. initi initialises each new instance of the class
        super().__init__(config) #call the constructor, aka the init of the parent class

        self.name = name #create an ATTRIBUTE called name and assign to it the value of the name parameter
        self.age = age #these are instance attributes
        self.batch_size = self.config.train.batch_size

    # #instance method: functions defined inside a class thta can only be called from an instance of that class
    # def description(self):
    #     return f"{self.name} is {self.age} years old"

    # def speak(self):
    #     return f"{self.name} says {sound}"

    # def __str__(self):
    #     return f"{self.name} is a very good boy who is {self.age} years old"

    # def __repr__(self):
    #     return f"{self.name} is a very good boy in the repr who is {self.age} years old"