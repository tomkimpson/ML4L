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
        #Process the raw ERA data
        raw_data_pipeline = ProcessERAData(CFG)
        raw_data_pipeline.process_time_constant_data()
        raw_data_pipeline.process_time_variable_data()

    if join_data:
        #Join the ERA and MODIS data together
        joining_method =  JoinERAWithMODIS(CFG)
        joining_method.join()


    #if allocate train/valid/test
    #COPY, dont move, raw data into working train/valid/test directories


    #create large files
    #how long do these pre processing steps take?
    #Are they necessary?


    #create small files

def train_and_predict(train_model):
    """Builds model, loads data, trains and evaluates"""
    NN = NeuralNet(CFG) #Create a NN using CFG configuration
    

    if train_model:

        NN.load_data()             # Load the training and validation data
        NN.construct_network()     # Construct and compile the network
        NN.train_network()         # Train it
        NN.save_model()            # Save the trained model



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process data and train a model')
    parser.add_argument('--process_data', dest='process_data', action='store_true',
                        help="Process raw ERA data")
    parser.add_argument('--join_data', dest='join_data', action='store_true',
                        help="Join MODIS and ERA data")
    parser.add_argument('--train_model', dest='train_model', action='store_true',
                        help="Train a model")
    return parser.parse_args()

if __name__ == '__main__':

    
    print ('-------------------------------')
    print ('-------------------------------')
    print ('-------------------------------')

    args = parse_arguments()
    options = vars(args)
    print('Running ML4L with arguments:')
    for k, v in options.items():
        print(k, v)
    print ('------------------------------')
    print ('-------------------------------')
    print ('-------------------------------')

    process_raw_data(args.process_data,args.join_data)
    train_and_predict(args.train_model)


    print ('------Completed OK---------')