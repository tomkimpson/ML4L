from configs.config import CFG
from model.neural_net import NeuralNet
from raw_data_processor.raw_data_processor import ProcessERAData
from raw_data_processor.join_ERA_with_MODIS import JoinERAWithMODIS
import argparse
 
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-a", "--archive", action="store_true", help="archive mode")
parser.add_argument("-v", "--verbose", action="store_true", help="increase verbosity")
parser.add_argument("-B", "--block-size", help="checksum blocksize")
parser.add_argument("--ignore-existing", action="store_true", help="skip files that exist")
parser.add_argument("--exclude", help="files to exclude")
parser.add_argument("src", help="Source location")
parser.add_argument("dest", help="Destination location")
args = parser.parse_args()
config = vars(args)
print(config)



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
    parser.add_argument('--joindata', dest='join_data', action='store_true',
                        help="Include CRPS/rank evaluation on full images")
    args = parser.parse_args()
    print(args)
    process_raw_data()


