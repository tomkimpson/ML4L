from configs.config import CFG
from model.neural_net import NeuralNet
from raw_data_processor.raw_data_processor import ProcessERAData
from raw_data_processor.join_ERA_with_MODIS import JoinERAWithMODIS
import argparse
 

def process_raw_data():

    """
    Get all data together and prepared to pass into model.
    This function should only have to be run once for a given set of data.
    Re-run when new data is introduced.
    """

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

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()
    print(args.accumulate(args.integers))



    process_raw_data()


