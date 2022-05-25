import pandas as pd
import sys
import xarray as xr
import numpy as np

#Global parameters

root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw' 

def process_month(dt,source): 
    
    
    if source == 'ERA_skin':
        name = '_skin_'
    if source == 'ERA_sfc':
        name = '_'
    
    #Open this month of data
    print('Open the dataset')
    d=dt.replace('-','_')
    fname = f'{root}/{source}/sfc{name}unstructured_{d}.grib'
    ds = xr.open_dataset(fname,engine='cfgrib',filter_by_keys={'typeOfLevel': 'surface'})
    print ('Loaded')

    #Get all variables/features
    all_variables = list(ds.keys())

    names = []
    variations = []
    for v in all_variables:
    
        a = ds[v].values
        #a == a[0,:] compares each value to the corresponding value in the first row
        #A column shares a common value if all the values in that column are True
        is_constant = all(np.all(a == a[0,:], axis = 0))
        
        
        print (v,is_constant)
        names.extend([v])
        variations.extend([is_constant])


    #Make it a df
    d = {'names': names, 'is_constant': variations}
    df = pd.DataFrame(data=d)
    df['dt'] = dt  
        
        
    return df
    

    
    

dates = pd.date_range('2018-01-01','2018-12-01', 
              freq='MS').strftime("%Y-%m").tolist()

print ('all_dates', dates)

def process_source(source,dates):
    
    print ('Running for: ', source)

    
    
    for dt in dates:
        print('Time = ', dt)
        df = process_month(dt,source)
        fname = f'small_data/booleanvariation_{dt}_{source}.pkl'
        print ('Saving file:', fname)
        df.to_pickle(fname)



    
process_source('ERA_skin',dates)
process_source('ERA_sfc',dates)
