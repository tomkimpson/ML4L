"""Abstract base model for Neural network"""

from abc import ABC,abstractmethod
from utils.config import Config 


class BaseModel(ABC):
    """Abstract model base class that is inherited by all child classes"""
    def __init__(self,cfg):
        self.config = Config.from_json(cfg)
        
        
    @abstractmethod
    def load_data(self):
        pass
