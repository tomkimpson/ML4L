from configs.config import CFG
from model.neural_net import NeuralNet
from raw_data_processor.raw_data_processor import ProcessERAData
from raw_data_processor.join_ERA_with_MODIS import JoinERAWithMODIS
import argparse
 

def process_raw_data(process_data, join_data):

    """
    Get all data together and prepared to pass into model.
    This function should only have to be run once for a given set of data.
    Re-run when new data is introduced.
    """

    if process_data:
        print ('process data')

    #raw_data_pipeline = ProcessERAData(CFG)
    
    #raw_data_pipeline.process_time_constant_data()
    #raw_data_pipeline.process_time_variable_data()

    if join_data:
        print('join data')
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

    parser = argparse.ArgumentParser(description='Process data and train a model')
    parser.add_argument('--process_data', dest='process_data', action='store_true',
                        help="Process raw ERA data")
    parser.add_argument('--join_data', dest='join_data', action='store_true',
                        help="Join MODIS and ERA data")
    

    args = parser.parse_args()
    print(args.process_data)



    process_raw_data(args.process_data,args.join_data)


