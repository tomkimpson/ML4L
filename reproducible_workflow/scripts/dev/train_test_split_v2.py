


import re

import glob


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def process_load(list_of_files,fname):

    print (fname)
    print('----------------------------------')
    dfs = []
    for f in list_of_files:
        print(f)
        df= pd.read_pickle(f)
        dfs.append(df)
        
        
    print('Concat')
    df = pd.concat(dfs)
    
    
    print('Writing HDF')
    fout = f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/{fname}.h5'
    print('Writing HDF')
    print(fout)
    df.to_hdf(fout,key=fname, mode='w')       
        


data_files= natural_sort(glob.glob(f'/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/V2matched_*.pkl'))

training_files = data_files[0:12]
validation_files = data_files[12:24]
test_files = data_files[24:36]


process_load(training_files,'training_data')
process_load(training_files,'validation_data')
process_load(training_files,'testing_data')
    