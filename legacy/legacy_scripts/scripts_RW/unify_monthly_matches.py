import re
import glob
import xarray as xr
import pandas as pd
import sys
import numpy as np



"""
Script that takes all the month files output bu join_MODIS_with-ERA.py and unifies
them into a single HDF.
"""



def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


data_files= natural_sort(glob.glob(f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/V2matched_*.pkl'))

dfs = []
for f in data_files[0:1]:
    print(f)
    df= pd.read_pickle(f)
    dfs.append(df)

print('Concat')
df = pd.concat(dfs)
print(df.columns)

#print ('Writing NPY')
#fout = f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/test_file.npy'
#npy_file = df.to_numpy()
#np.save(fout,npy_file)

print ('Writing CSV')
fout = f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/test_file.csv'
df.to_csv(fout)


#print('Writing HDF')
#fout = f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/all_months_V3.h5'
#df.to_hdf(fout, key='df', mode='w') 
#print('Done')















# for v in ['v15', 'v20']:
#     data_files= natural_sort(glob.glob(f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/{v}/matched_*.pkl'))
#     print('v:', v)
#     dfs = []
#     for f in data_files:
#         print(f)
#         df= pd.read_pickle(f)
#         dfs.append(df)
    
#     print('Concat')
#     df = pd.concat(dfs)

#     print('Writing HDF')
#     fout = f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/{v}/all_months.h5'
#     df.to_hdf(fout, key='df', mode='w') 
#     print('Done')



