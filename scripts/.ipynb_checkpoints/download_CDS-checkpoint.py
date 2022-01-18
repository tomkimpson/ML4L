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
    cds_xarray = xdata.to_xarray(backend_kwargs={'errors': 'ignore','filter_by_keys':{'edition': 1, 'typeOfLevel':'surface'}})
    cds_xarray.to_netcdf(data_root+xdata)
else:
    cds_xarray = xr.open_dataset(data_root+xdata)
