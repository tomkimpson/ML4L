# -*- coding: utf-8 -*-
"""Model config in json format"""

root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/' 

CFG = {
    "data": {

        #Paths to raw data
        "path_to_raw_ERA_skin": '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/ERA_skin/',
        "path_to_raw_ERA_sfc": '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/ERA_sfc/',
        "path_to_raw_ERA_skt": '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/ERA_skt/',
        "path_to_raw_V15_climate_fields": f'{root}climate.v015/climate.v015/639l_2/',
        "path_to_raw_V20_climate_fields": f'{root}climate.v020/climate.v020/639l_2/',
        "path_to_monthly_clake_files": '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/BonusClimate/',
        "path_to_MODIS_data": '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/MODIS',


         #Edge cases to handle in ERA_skin
        "ERA_skin_constant_features": ['slt','sdfor'],                                   # These are the features in ERA_skin that are constant, but are not in the V* climate files
        "ERA_skin_variable_features": 'aluvp/aluvd/alnip/alnid/istl1/istl2/sd/2d/fal',   # These are the features in ERA_skin that are not constant
        
        #Paths to processed output data
        "path_to_processed_V15_climate_fields": f'{root}processed_data/ERA_timeconstant/ERA2_constants_v15.nc',
        "path_to_processed_V20_climate_fields": f'{root}processed_data/ERA_timeconstant/ERA2_constants_v20.nc',
        "path_to_processed_variable_fields": f'{root}processed_data/ERA_timevariable/',
        "path_to_joined_ERA_MODIS_files": f'{root}/joined_data/'

        #Parameters for processing raw data
        "tmpdir": f'{root}tmpdir',
        "min_year":2018,
        "max_year":2020,
         
        #Parameters for joining ERA/MODIS
        "aquaDay_min_hour":2,
        "terraDay_min_hour":-1,
        "aquaNight_min_hour":-1,
        "terraNight_min_hour":11,
 
        "aquaDay_max_hour":24,
        "terraDay_max_hour":22,
        "aquaNight_max_hour":13,
        "terraNight_max_hour":24,


        "aquaDay_local_solar_time":"13:30",
        "terraDay_local_solar_time":"10:30",
        "terraNight_local_solar_time":"22:30",
        "aquaNight_local_solar_time":"01:30",
              
        "satellite":'aquaDay',
        "latitude_bound": 70,

        "training_data": '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/test_data.tfrecords',
        "validation_data": '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/test_data.tfrecords'

    },
    "train": {
        "batch_size": 64,
        "buffer_size": 1000,
        "epoches": 20,
        "val_subsplits": 5,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3
    }
}


