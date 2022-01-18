import xarray as xr
from config import *




#Load the data
cds_xarray = xr.open_dataset(data_root+"xdata.nc")

#Select timestamp
new = cds_xarray.sel(time='2017-01-01')

#print (cds_xarray)
print(new)