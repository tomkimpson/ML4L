import xarray as xr
import glob
import sys

root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/playground/'

#Path to directory of ERA files
ERA_path = root+'ERA_test/'

#Path to directory of MODIS files
MODIS_path = root+'MODIS_test/'




def load_ERA_data(f):
    """Load ERA data and convert to long1"""

    print ('LOADING', f)
    ds_ERA = xr.open_dataset(f, engine="cfgrib")
    print ('LOADED')
    df_ERA = ds_ERA.to_dataframe()
    df_ERA['long1'] = (df_ERA['longitude'] +180.0) %360.0 - 180.0
    
    print('resetting index')
    df_ERA = df_ERA.reset_index().set_index(['latitude','long1', 'time'])

    
    return df_ERA

def load_MODIS_data(f):
    """Open xarray ds, convert to pandas df and make some corrections"""
    
    #Open
    ds = xr.open_rasterio(f)
    
    #Turn it into a df, and make some basic corrections
    df = ds.to_dataframe(name='local_solar_time_uncorrected')
    
    #Correct local solar time and calculate UTC
    df['local_solar_time'] = df['local_solar_time_uncorrected']/0.02*0.1 #scaling
    df['UTC'] = df['local_solar_time'] - df.index.get_level_values('x')/15.0 #Correct LST to UTC
    
    
    #---Dates
    #Get the date from string name
    date_string =f.split('_')[-1].split('.')[0].replace('-','')
    date = pd.to_datetime(date_string, format='%Y%m%d')
    df['date'] = date_string
    
    #Get a datetime by combining date and UTC, and round to hours
    df['timeUTC'] = date + pd.to_timedelta(df['UTC'], unit='h') 
    df['time']=df['timeUTC'].dt.round('H')
    
    out = df.reset_index().set_index(['y', 'x', 'time'],drop=True) #reindex()
    return out



ERA_files = sorted(glob.glob(ERA_path + '*'))
MODIS_files = sorted(glob.glob(MODIS_path + '*'))

f = ERA_files[0]
g = MODIS_files[0]

print (f)
print(g)

#Load data, convert to df, make some corrections
df_ERA = load_ERA_data(f)
df_MODIS=load_MODIS_data(g)




