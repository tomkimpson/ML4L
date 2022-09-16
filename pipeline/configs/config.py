# -*- coding: utf-8 -*-
"""Model config in json format"""

root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/' 

CFG = {
    "data": {

        #A. Paths to raw data
        "path_to_raw_ERA_skin":           f'{root}ERA_skin/',
        "path_to_raw_ERA_sfc":            f'{root}ERA_sfc/',
        "path_to_raw_ERA_skt":            f'{root}ERA_skt_tom/',
        "path_to_raw_V15_climate_fields": f'{root}climate.v015/climate.v015/639l_2/',
        "path_to_raw_V20_climate_fields": f'{root}climate.v020/climate.v020/639l_2/',
        "path_to_monthly_clake_files":    f'{root}BonusClimate/',
        "path_to_saline_clake_files":     f'{root}saltlakes_max/clake_639l2_yearMAX_saline',
        "path_to_MODIS_data":             f'{root}MODIS',


        #B. Paths to processed output data
        "path_to_processed_V15_climate_fields": f'{root}processed_data/ERA_timeconstant/ERA_constants_v15.nc',
        "path_to_processed_V20_climate_fields": f'{root}processed_data/ERA_timeconstant/ERA_constants_v20.nc',
        "path_to_processed_variable_fields":    f'{root}processed_data/ERA_timevariable/',
        "path_to_joined_ERA_MODIS_files":       f'{root}processed_data/joined_data/',


        # C. Edge cases to handle in ERA_skin
        "ERA_skin_constant_features": ['slt','sdfor'],                                   # These are the features in ERA_skin that are constant, but are not in the V* climate files
        "ERA_skin_variable_features": 'aluvp/aluvd/alnip/alnid/istl1/istl2/sd/2d/fal',   # These are the features in ERA_skin that are not constant
        
        # D. Parameters for processing raw data
        "min_year_to_process":2015, 
        "max_year_to_process":2015,
         
        # E. Parameters for joining ERA/MODIS
        "min_year_to_join": 2021, # typically we want these min/maxes to be the same as min/max_year_to_process, but maybe an edge case exists?
        "max_year_to_join": 2021,
        "joining_metric" : 'haversine', #L2, haversine
        "IO_prefix": 'RapidsHaversine_MODIS_', #{year_month}.parquet'
        

        ##Satellite timings
        "satellite":'aquaDay',
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
        "latitude_bound": 70,


        #F. Parameters for creating ML input files
        "training_years":   ['2016'],
        "validation_years": ['2017'],
        "test_years":       ['2019'],

        ## List of all features, grouped
        "list_of_meta_features": ['latitude_ERA', 'longitude_ERA','time','number_of_modis_observations'],

        # "list_of_time_variable_features" : ['sp', 'msl', 'u10', 'v10', 't2m', 'aluvp', 'aluvd',
        #                                     'alnip', 'alnid', 'istl1', 'istl2', 'sd', 'd2m', 'fal', 'skt'],

        "list_of_time_variable_features" : ['sp', 'msl', 'u10', 'v10', 't2m', 'aluvp', 'aluvd',
                                            'alnip', 'alnid', 'istl1', 'istl2', 'sd', 'd2m', 'fal','MODIS_LST'],

        "list_of_V15_features":            ['slt_v15', 'sdfor_v15', 'sdor_v15', 'lsrh_v15', 'cvl_v15', 'sr_v15',
                                            'lsm_v15', 'isor_v15', 'tvl_v15', 'tvh_v15', 'cvh_v15', 'si10_v15',
                                            'anor_v15', 'cl_v15', 'dl_v15', 'z_v15', 'slor_v15'],

        "list_of_V20_features":            ['slt_v20','sdfor_v20', 'sdor_v20', 'lsrh_v20', 'cvl_v20', 'sr_v20', 
                                            'lsm_v20','isor_v20', 'tvl_v20', 'tvh_v20', 'cvh_v20', 'si10_v20', 
                                            'anor_v20','cl_v20', 'dl_v20', 'z_v20', 'slor_v20'], 
                                
                                
        "list_of_bonus_features":           ['clake_monthly_value','cl_saline'], 
        "target_variable" :                 ["skt"],
    },
    "train": {
        "training_data":   f'{root}processed_data/joined_data/2016_MLS.parquet',
        "validation_data": f'{root}processed_data/joined_data/2017_MLS.parquet',
        "training_features": ['sp', 'msl', 'u10', 'v10', 't2m','aluvp', 'aluvd',
                              'alnip', 'alnid', 'istl1', 'istl2', 'sd','d2m', 'fal','skt',
                              'slt_v15', 'sdfor_v15', 'sdor_v15', 'cvl_v15','lsm_v15', 'isor_v15', 
                              'tvl_v15', 'tvh_v15', 'cvh_v15', 'si10_v15','anor_v15', 'cl_v15', 'dl_v15', 
                              'z_v15', 'slor_v15',
                            #   'sdor_v20', 'cvl_v20','lsm_v20', 'isor_v20', 
                            #    'cvh_v20', 'si10_v20','anor_v20', 'cl_v20', 'dl_v20', 
                            #    'z_v20', 'slor_v20',
                               'clake_monthly_value','cl_saline'
                              ], #Of all the available features in the training data, which should be used? It is known that DELTA ['lsrh_v15' 'sr_v15' 'slt_v20' 'sdfor_v20' 'lsrh_v20' 'sr_v20' 'tvl_v20' 'tvh_v20'] are all the same
        "batch_size":              1024,
        "epochs":                  300,
        "number_of_hidden_layers": 4,
        "nodes_per_layer":         [None,None,None,None],
        "target_variable":         'MODIS_LST',
        "learning_rate":           3e-4,
        "loss":                    'mse',
        "metrics":                 ["accuracy"],
        "path_to_trained_models":  f'{root}processed_data/trained_models/',
        "model_name":              'V15_2016_X_V5', #This model will also be used for prediction
        "use_pretrained_model":    False,
        "overwrite":               False,
        "epoch_save_freq":         10,
        "early_stopping_patience": 20,
    },

    "predict": {
        "testing_data":    f'{root}processed_data/joined_data/2019_MLS.parquet',
        "testing_data_query": None # "abs(cl_v20) > 0.1" #This query will be applied to the testing data after loading                                      
    },
    "permute" : {
        "features_to_permute": 'ALL' #['cvl_v20', 'cvh_v20'] # Either ['A', 'B',...] or 'ALL'
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


