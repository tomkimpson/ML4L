


import xarray as xr



root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/'
f = 'ml_skin_unstructured_2010_01.grib'
f2 = 'sfc_skin2_unstructured_2010_01.grib'
f3 = 'TEMP.grib'


ds_grib = xr.open_dataset(root+f, engine="cfgrib")
print (ds_grib)

