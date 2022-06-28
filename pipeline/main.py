from configs.config import CFG
from model.neural_net import NeuralNet
from raw_data_processor.raw_data_processor import ProcessERAData
from raw_data_processor.join_ERA_with_MODIS import JoinERAWithMODIS
from raw_data_processor.prep_for_ML import PrepareMLData

import argparse
 

def process_raw_data(process_data, join_data,greedy):

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


    if greedy:
        prep = PrepareMLData(CFG) 
        prep.greedy_preprocessing()




def train_and_predict(train_model,predict,evaluate):
    """Builds model, loads data, trains and evaluates"""
    
    NN = NeuralNet(CFG)        # Create a NN using CFG configuration

    if train_model:
        NN.train()
        
    if predict:
        NN.predict()

    if evaluate:
        NN.evaluate()



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process data and train a model')
    parser.add_argument('--process_raw_data', dest='process_data', action='store_true',
                        help="Process raw ERA data")
    parser.add_argument('--join_era_modis', dest='join_data', action='store_true',
                        help="Join MODIS and ERA data")
    parser.add_argument('--ML_prep', dest='prep_ML', action='store_true',
                        help="Prepare joined data to be fed into training model")
    parser.add_argument('--train_model', dest='train_model', action='store_true',
                        help="Train a model")
    parser.add_argument('--predict', dest='predict', action='store_true',
                        help="Use a trained model to make some predictions")
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help="Evaluate a trained model on test data and feature importance")
    return parser.parse_args()

if __name__ == '__main__':

    
    print ('-------------ML4L---------------')
    print ('-------------------------------')
    print ('-------------------------------')

    args = parse_arguments()
    options = vars(args)
    print('Running with arguments:')
    for k, v in options.items():
        print(f'{k : <15} {v}')
        
    print ('------------------------------')
    print ('-------------------------------')
    print ('-------------------------------')

    process_raw_data(args.process_data,args.join_data,args.prep_ML)
    train_and_predict(args.train_model,args.predict,args.evaluate)


    print ('------Completed OK---------')