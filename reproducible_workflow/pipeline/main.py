from configs.config import CFG
from model.neural_net import NeuralNet
from raw_data_processor.raw_data_processor import ProcessERAData,JoinERAWithMODIS


def process_raw_data():

    #raw_data_pipeline = ProcessERAData(CFG)
    
    #raw_data_pipeline.process_time_constant_data()
    #raw_data_pipeline.process_time_variable_data()

    joining_method =  JoinERAWithMODIS(CFG)
    joining_method.join()
    
def run():
    """Builds model, loads data, trains and evaluates"""
    model = NeuralNet(CFG) #Create a NN using CFG configuration
    model.load_data()
    
    print(model.training_data)
    #model.build()
    #model.train()
    #model.evaluate()


if __name__ == '__main__':
    process_raw_data()


#print(buddy)

#https://theaisummer.com/best-practices-deep-learning-code/