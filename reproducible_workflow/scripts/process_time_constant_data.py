import os
import xarray as xr
import shutil
import glob

"""
Script to process the time-constant ERA5 features and bring them together into a single dataset.
This is done for both V15 ad V20 data, so the output is two separate files.
See Workflow/ 1.1 ERA5 Data
"""



#Specify root path
root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/' 

#Create a tmp directory where we will write to
tmpdir = root+'tmp/'
if os.path.exists(tmpdir):
    shutil.rmtree(tmpdir)
os.mkdir(tmpdir)

#Select just a single ERA file, and the constant features you want from that file
ERA_skin = root + '/ERA_skin/sfc_skin_unstructured_2018_01.grib'
constant_features = ['slt','sdfor'] #These are the features in ERA_skin that ARE constant, but are not in V* files


#Load that file, select the data slice you want #, save it to disk
ds = xr.open_dataset(ERA_skin,engine='cfgrib',filter_by_keys={'typeOfLevel': 'surface'},backend_kwargs={'indexpath': ''}) #Assumes constant features are surface quantities, which is currently true, but may not always be...
selected_data = ds[constant_features].isel(time=[0])


#Now deal with the separate V15/V20 surface fields
climate_v15_path = root+'climate.v015/climate.v015/639l_2/'
climate_v20_path = root+'climate.v020/climate.v020/639l_2/'

outfile_v15 = root + 'processed_data/ERA_timeconstant/ERA_constants_v15.nc'
outfile_v20 = root + 'processed_data/ERA_timeconstant/ERA_constants_v20.nc'

input_paths = [climate_v15_path,climate_v20_path]
output_paths = [outfile_v15,outfile_v20]
for i in range(len(input_paths)):  
    path = input_paths[i]
    outpath = output_paths[i]
    version_files = glob.glob(path+'*') #Get all the grib files in that directory
    splitfile = tmpdir+'/splitfile_[shortName].grib'
    
    for f in version_files: #Split each file into sub-files by feature
        query_split = f'grib_copy {f} "{splitfile}"' 
        os.system(query_split)
        
    splitfiles = glob.glob(tmpdir + '*.grib') #Get a list of all the splitfiles you have just created
    ds_all = []
    ds_all.append(selected_data) #Add the data slice you took above
    for f in splitfiles: #Load each file in turn
        ds = xr.open_dataset(f,engine='cfgrib',backend_kwargs={'indexpath': ''})
        ds_all.append(ds)


    constant_merged_data = xr.merge(ds_all,compat='override') #need the override option to deal with keys
    constant_merged_data.to_netcdf(outpath) #write to disk
    print(outpath)
    
