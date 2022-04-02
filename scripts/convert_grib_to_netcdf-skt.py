import xarray as xr
import pandas as pd



def process_grib_file(f,output_path):
    
    #Open file
    print ('Loading file')
    ds = xr.open_dataset(f,engine='cfgrib',filter_by_keys={'typeOfLevel': 'surface'},backend_kwargs={'indexpath': ''})
    
    print('Loaded')
    #Relabel longitude coordinate to be consistent with MODIS
    ds = ds.assign_coords({"longitude": (((ds.longitude + 180) % 360) - 180)})
    
    
    #Group it by time 
    ds_grouped = ds.groupby("time")
    
    
    
    #Output path
    

    for label,group in ds_grouped:    
        outname = output_path+str(label)+'.nc'
        print(outname)
        group.to_netcdf(outname)

    #Explictly close everything
    ds.close()
    del ds_grouped
    
    
    


            
#Paths
root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw' 

    
#Parameters
dates = pd.date_range('2018-01-01','2020-12-01', 
              freq='MS').strftime("%Y-%m").tolist()



source = 'ERA_skt'
out = f'{root}/ERA_skt_netcdf/'

for dt in dates:
    d=dt.replace('-','_')
    
    
    
    fname = f'{root}/{source}/skt_unstructured_{d}.grib'
   

    print('Processing month:', fname)
    
    process_grib_file(fname,out)
    
 
    
