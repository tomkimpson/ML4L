import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-land-monthly-means',
    {
        'format': 'grib',
        'variable': [
            'lake_mix_layer_temperature', 'snow_depth_water_equivalent', 'snowmelt',
            'soil_temperature_level_2',
        ],
        'year': [
            '2017', '2018'
        ],
        'month': [
            '01', '02', '03',
        ],
        'time': '00:00',
    },
    'download.grib')


# #Load the data
# if load_x_data_from_remote:
#     xdata = cml.load_source("cds",
#                             "reanalysis-era5-land-monthly-means",
#                              variable=variables,
#                              product_type= "monthly_averaged_reanalysis",
#                              year = years,
#                              month = months,
#                              time = times
#                              )
#     cds_xarray = xdata.to_xarray(backend_kwargs={'errors': 'ignore','filter_by_keys':{'edition': 1, 'typeOfLevel':'surface'}})
#     cds_xarray.to_netcdf(data_root+"xdata.nc")
# else:
#     cds_xarray = xr.open_dataset(data_root+"xdata.nc")
