

import pandas as pd
#import json


input_file = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/MODIS_ERA_joined_data_averaged.pkl'
train_condition = pd.to_datetime("2019-01-01 00:00:00")
test_condition  = pd.to_datetime("2020-01-01 00:00:00")
epochs = 100
batch_size = 100000







parameters_dict = {'input_file':     input_file,
                  'train_condition': train_condition,
                  'test_condition':  test_condition,
                  'epochs':          epochs,
                  'batch_size':      batch_size}



with open('meta.txt', 'w') as f:
        for k,v in parameters_dict.items():
            row = k + ": " + str(v) + "\n"
            f.write(row)
