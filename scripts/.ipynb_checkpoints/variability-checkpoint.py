import pandas as pd
import sys
import xarray as xr


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
        variance = ds[v].std(dim='time') #standard deviation over entire time period (1 month) per grid point
        mean = ds[v].mean(dim='time') #mean value over entire time period (1 month) per grid point
        CV = variance/mean #coefficient of variation
        av_CV = CV.mean() #average of the coefficient of variation over all grid points
    
        print (av_CV.name,float(av_CV.data))
        
        names.extend([av_CV.name])
        variations.extend([float(av_CV.data)])


    #Make it a df
    d = {'names': names, 'CV': variations}
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
        fname = f'small_data/variation_{dt}_{source}.pkl'
        print ('Saving file:', fname)
        df.to_pickle(fname)



    
#process_source('ERA_skin',dates)
process_source('ERA_sfc',dates)
