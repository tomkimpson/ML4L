from re import A
import sys
import glob
from turtle import fd
import pandas as pd




"""A script to take monthly data in separate train/validate/test folders,
    and produce a single file for each train/validate/test.
    We also caculate the "delta features" i.e. V20 - V15 for the time constant features
    All features are normalised w.r.t. the training set parameters.
    For the test set we also carry latitude/longitude/position.
    Target variable, MODIS_LST is not normalised.
"""



root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/'
output_directory = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/'






#Manually select which columns are "global" information (e.g. lat/long).
#These features will not be normalised and will not be part of the training or validation sets.
global_information = ['latitude_ERA', 'longitude_ERA','time'] 

#Training features, will be normalised
time_variable_features = ['sp', 'msl', 'u10', 'v10', 't2m', 
                          'aluvp', 'aluvd','alnip', 'alnid', 'istl1', 'istl2', 'sd', 'd2m', 'fal', 
                          'skt']

time_constant_features = ['lsm_v15','cl_v15','dl_v15','cvh_v15','cvl_v15',
                          'anor_v15','isor_v15','slor_v15','sdor_v15','sr_v15','lsrh_v15',
                          'si10_v15'] #these are the constant fields, V15. Will also calculate delta corrections of these fields.

#Target, will not be normalised
target_variable = ['MODIS_LST']


def calculate_delta_fields(df,fields):
    
    """Function to determine V20 - V15 for different time constant fields"""
      
    new_column_names = []
    for i in fields:
        feature = i.split('_')[0] #cl_v15 --> cl
        column_name = f'{feature}_delta'
        new_column_names.append(column_name)
        v20_name = feature+'_v20'
        df[column_name] = df[v20_name] - df[i]
                
    return new_column_names,df      




def process_directory(d,n1,n2,unnormalised_features):
    print ('Loading directory:', d)
    data_files = glob.glob(root+d+'/*')

    dfs = []
    for f in data_files:
        print (f)

        #Load the monthly file
        df= pd.read_pickle(f)

        #Calculate extra "delta" columns V20 - V15
        delta_fields,df = calculate_delta_fields(df,time_constant_features)

        #Don't select all the columns
        selected_columns = unnormalised_features + time_variable_features + time_constant_features + delta_fields
        selected_df = df[selected_columns] 

        dfs.append(selected_df)

    print('All files processed. Now concat')
    df = pd.concat(dfs)

    print ('Split into data which will be normalised')
    df_meta = df[unnormalised_features]
    df_features = df.drop(columns=unnormalised_features, axis=1)

    print(df_meta.columns)
    print(df_features.columns)

    
    if n1 is None:
        print ('Calculating normalisation parameters')
        #If we dont have any normalisation parameters already 
        normalisation_mean =  df_features.mean()
        normalisation_std =  df_features.std()

    else:
        normalisation_mean = n1
        normalisation_std = n2 


    #Normalise it using the pre calculated terms
    df_features = (df_features-normalisation_mean)/normalisation_std

    #Create new df composed of the unnormalsied meta information and the normalised features 
    df = pd.concat([df_meta,df_features],axis=1)

    #Write to disk 
    print('Writing HDF')
    df.to_hdf(output_directory + d +'.h5', key='df', mode='w') 


    return normalisation_mean, normalisation_std





normalisation_mean, normalisation_std = process_directory('training_data',None,None,target_variable)
process_directory('validation_data',normalisation_mean,normalisation_std,target_variable)
process_directory('test_data',normalisation_mean,normalisation_std,target_variable+global_information)



print('Done')















    
    
    
    
    
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
