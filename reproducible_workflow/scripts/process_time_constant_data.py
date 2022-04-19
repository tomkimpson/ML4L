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


###---------------------------------------------------###
###----Extract time constant features from ERA_skin---###
###---------------------------------------------------###
ERA_skin = root + '/ERA_skin/sfc_skin_unstructured_2018_01.grib' #Select just a single ERA file, 
ds = xr.open_dataset(ERA_skin,engine='cfgrib',filter_by_keys={'typeOfLevel': 'surface'},backend_kwargs={'indexpath': ''}) #Load it, assumes constant features are also surface quantities
constant_features = ['slt','sdfor'] #These are the features in ERA_skin that ARE constant, but are not in V* files
selected_data = ds[constant_features].isel(time=[0])




###---------------------------------------------------###
###----Extract time constant features from yearlyCL---###
###---------------------------------------------------###
yearlyCL =   root+'BonusClimate/yearlyCL/clake' #Bonus clake data which is constant in time
ds_bonus = xr.open_dataset(yearlyCL,engine='cfgrib',decode_times = False,backend_kwargs={'indexpath': ''})



###---------------------------------------------------###
###----------Deal with V15/V20 surface fields---------###
###---------------------------------------------------###

#Method here, for a particular version, is to:
#.   Get all the climate files
#.   Split by parameter to produce a bunch of files
#.   Load each new file, append it to array
#.   Append in the two datasets from above, ERA_sfc and yearlyCL
#.   Merge it all together


#input_path: output_path
climateV = {root+'climate.v015/climate.v015/639l_2/':root + 'processed_data/ERA_timeconstant/ERA_constants_v15.nc',
            root+'climate.v020/climate.v020/639l_2/':root + 'processed_data/ERA_timeconstant/ERA_constants_v20.nc'
           }



for v in climateV:
    input_path = v
    output_path = climateV[v]
    version_files = glob.glob(input_path+'*')              # Get all the grib files in that directory
    splitfile = tmpdir+'/splitfile_[shortName].grib'
    
    for f in version_files: #Split each file into sub-files by feature
        query_split = f'grib_copy {f} "{splitfile}"' 
        os.system(query_split)
        

    ds_all = []                               # Create an empty array
    ds_all.append(selected_data)              # Add the data slice you took above
    ds_all.append(ds_bonus)                   #...and the constant bonus data
    
    
    splitfiles = glob.glob(tmpdir + '*.grib') # Get a list of all the splitfiles you have just created
    for f in splitfiles: #Load each file in turn
        ds = xr.open_dataset(f,engine='cfgrib',backend_kwargs={'indexpath': ''})
        ds_all.append(ds)


    constant_merged_data = xr.merge(ds_all,compat='override') #need the override option to deal with keys
    constant_merged_data.to_netcdf(output_path,mode='w') #write to disk
    print(output_path)
    
