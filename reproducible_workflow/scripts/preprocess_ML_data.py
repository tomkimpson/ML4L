import sys
import glob

import pandas as pd

root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/'

directories = ['training_data/','validation_data/','test_data/']



#Grab the training data, normalise it all
d = 'training_data/'
selected_files = glob.glob(root+d+'*')

#
list_of_all_features = []
list_of_all_meta_information = []
list_of_all_features_to_drop

for f in selected_files:
    df= pd.read_pickle(f)
    print(df)
    print(df.columns)
    sys.exit()
    print(i)

# for d in directories:
#     selected_files = glob.glob(root+d+'*')
#     for i in selected_files:
#         print(i)

# filenames = 'V2matched_[0-9].pkl'
# training_files = glob.glob(filenames)



# core_features = ['sp', 'msl', 'u10', 'v10', 't2m', 
#                  'aluvp', 'aluvd','alnip', 'alnid', 'istl1', 'istl2', 'sd', 'd2m', 'fal', 
#                  'skt'] #these are the time variable fields
# surface_features = ['lsm_v15','cl_v15','dl_v15','cvh_v15','cvl_v15',
#                     'anor_v15','isor_v15','slor_v15','sdor_v15','sr_v15','lsrh_v15',
#                     'si10_v15'] #these are the constant fields, V15


    
# feature_names = core_features+surface_features
# target_variable = ['MODIS_LST'] #The variable you are trying to learn/predict
# selected_columns = feature_names+target_variable







# dfs = []
# for f in training_files:
#     print(f)
#     df= pd.read_pickle(f)
#     df_selected = df[selected_columns]
#     dfs.append(df_selected)
        
        
#     print('Concat')
#     df = pd.concat(dfs)
    
    
    
    
    
# Index(['latitude_ERA', 'longitude_ERA', 'latitude_MODIS', 'longitude_MODIS',
#        'MODIS_LST', 'sp', 'msl', 'u10', 'v10', 't2m', 'aluvp', 'aluvd',
#        'alnip', 'alnid', 'istl1', 'istl2', 'sd', 'd2m', 'fal', 'skt',
#        'slt_v15', 'sdfor_v15', 'vegdiff_v15', 'lsrh_v15', 'cvh_v15',
#        'isor_v15', 'dl_v15', 'lsm_v15', 'z_v15', 'si10_v15', 'sdor_v15',
#        'cvl_v15', 'anor_v15', 'slor_v15', 'sr_v15', 'tvh_v15', 'tvl_v15',
#        'cl_v15', 'slt_v20', 'sdfor_v20', 'vegdiff_v20', 'lsrh_v20', 'cvh_v20',
#        'isor_v20', 'dl_v20', 'lsm_v20', 'z_v20', 'si10_v20', 'sdor_v20',
#        'cvl_v20', 'anor_v20', 'slor_v20', 'sr_v20', 'tvh_v20', 'tvl_v20',
#        'cl_v20', 'COPERNICUS/', 'CAMA/', 'ORCHIDEE/',
#        'monthlyWetlandAndSeasonalWater_minusRiceAllCorrected_waterConsistent/',
#        'CL_ECMWFAndJRChistory/', 'heightAboveGround', 'L2_distance',
#        'H_distance', 'time'],
#       dtype='object')