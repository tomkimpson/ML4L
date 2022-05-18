# -*- coding: utf-8 -*-
"""Model config in json format"""

root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/' 

CFG = {
    "data": {

        #Paths to raw data
        "path_to_raw_ERA_skin":           f'{root}ERA_skin/',
        "path_to_raw_ERA_sfc":            f'{root}ERA_sfc/',
        "path_to_raw_ERA_skt":            f'{root}ERA_skt_tom/',
        "path_to_raw_V15_climate_fields": f'{root}climate.v015/climate.v015/639l_2/',
        "path_to_raw_V20_climate_fields": f'{root}climate.v020/climate.v020/639l_2/',
        "path_to_monthly_clake_files":    f'{root}BonusClimate/',
        "path_to_saline_clake_files":     f'{root}saltlakes/clake_639l2_year_saline',
        "path_to_MODIS_data":             f'{root}MODIS',


        # Edge cases to handle in ERA_skin
        "ERA_skin_constant_features": ['slt','sdfor'],                                   # These are the features in ERA_skin that are constant, but are not in the V* climate files
        "ERA_skin_variable_features": 'aluvp/aluvd/alnip/alnid/istl1/istl2/sd/2d/fal',   # These are the features in ERA_skin that are not constant
        
        #Paths to processed output data
        "path_to_processed_V15_climate_fields": f'{root}processed_data/ERA_timeconstant/ERA2_constants_v15.nc',
        "path_to_processed_V20_climate_fields": f'{root}processed_data/ERA_timeconstant/ERA2_constants_v20.nc',
        "path_to_processed_variable_fields":    f'{root}processed_data/ERA_timevariable/',
        "path_to_joined_ERA_MODIS_files":       f'{root}processed_data/joined_data/',

        #Parameters for processing raw data
        "min_year_to_process":2016, 
        "max_year_to_process":2021,
         
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

        "min_year_to_join":2020, #typically we want these min/maxes to be the same as min_year_to_join, but maybe an edge case exists?
        "max_year_to_join":2020,


       
        "joining_metric" : 'haversine', #L2, haversine



        "training_years": ['2018'],
        "validation_years": ['2019'],
        "test_years":['2020'],

        #"path_to_training_data":   f'{root}processed_data/joined_data/dev_train/',
        #"path_to_validation_data": f'{root}processed_data/joined_data/dev_valid/',
        #"path_to_test_data": f'{root}processed_data/joined_data/dev_test/',

        "list_of_meta_features": ['latitude_ERA', 'longitude_ERA','time'],
        "list_of_time_variable_features" : ['sp', 'msl', 'u10', 'v10', 't2m', 'aluvp', 'aluvd',
                                            'alnip', 'alnid', 'istl1', 'istl2', 'sd', 'd2m', 'fal', 'skt'],
        "list_of_V15_features": ['slt_v15', 'sdfor_v15', 'sdor_v15', 'lsrh_v15', 'cvl_v15', 'sr_v15',
                                 'lsm_v15', 'isor_v15', 'tvl_v15', 'tvh_v15', 'cvh_v15', 'si10_v15',
                                 'anor_v15', 'cl_v15', 'dl_v15', 'z_v15', 'slor_v15'],

        "list_of_V20_features": ['slt_v20','sdfor_v20', 'sdor_v20', 'lsrh_v20', 'cvl_v20', 'sr_v20', 
                                'lsm_v20','isor_v20', 'tvl_v20', 'tvh_v20', 'cvh_v20', 'si10_v20', 
                                'anor_v20','cl_v20', 'dl_v20', 'z_v20', 'slor_v20'], 
                                
                                
        "list_of_bonus_features": ['clake_monthly_value','cl_saline'], 
        "target_variable" : ["MODIS_LST"],

    },
    "train": {
        "training_data": f'{root}processed_data/joined_data/2018_ML.parquet',
        "validation_data": f'{root}processed_data/joined_data/2019_ML.parquet',
        "testing_data": f'{root}processed_data/joined_data/2020_ML.parquet',
        "training_features": ['sp', 'msl', 'u10', 'v10', 't2m', 'aluvp', 'aluvd',
                              'alnip', 'alnid', 'istl1', 'istl2', 'sd', 'd2m', 'fal', 'skt',
                              'lsm_v15','cl_v15','dl_v15','cvh_v15','cvl_v15',
                              'anor_v15','isor_v15','slor_v15','sdor_v15','sr_v15','lsrh_v15',
                              'si10_v15',
                              'lsm_v20','cl_v20','dl_v20','cvh_v20','cvl_v20',
                              'anor_v20','isor_v20','slor_v20','sdor_v20','sr_v20','lsrh_v20',
                              'si10_v20'
                              ],
        "batch_size": 10000,
        "epochs": 1,
        "number_of_hidden_layers":2,
        "nodes_per_layer": [19,19],
        "target_variable": 'MODIS_LST',
        "learning_rate": 3e-4,
        "loss": 'mse',
        "metrics": ["accuracy"],
        "path_to_trained_models": f'{root}processed_data/trained_models/',
        "model_name": 'V20_2018_testing',
        "overwrite": False,
        "use_pretrained_model":False,
        "epoch_save_freq": 10,
        "early_stopping_patience":20
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


