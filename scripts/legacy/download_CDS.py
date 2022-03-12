import climetlab as cml
import xarray as xr
from config import *



#Load the data
if load_x_data_from_remote:
    xdata = cml.load_source("cds",
                            "reanalysis-era5-land-monthly-means",
                             variable=list(xvariables.values()),
                             product_type= "monthly_averaged_reanalysis",
                             year = years,
                             month = months,
                             time = times
                             )

    print ('DATA loaded from cache')
    print (xdata)

    print('--------')
    print ('Now trying to load into cube')
#    print(xdata.to_xarray(xarray_open_mfdataset_kwargs = {'filter_by_keys':{'typeOfLevel': 'newsurface'}},backend_kwargs={'errors': 'ignore', 'filter_by_keys': {'typeOfLevel': 'notsurface'}}))
    print(xdata.to_xarray(backend_kwargs={'errors': 'ignore', 'filter_by_keys': {'typeOfLevel': 'notsurface'}}))
 #   data = xdata.to_xarray(engine='cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel':'surface', 'edition': 1}})


    print('--------')



    cds_xarray = xdata.to_xarray(backend_kwargs={'errors': 'ignore','filter_by_keys':{'edition': 1, 'typeOfLevel':'surface'}})
    cds_xarray.to_netcdf(data_root+xdata)
else:
    cds_xarray = xr.open_dataset(data_root+xdata)
