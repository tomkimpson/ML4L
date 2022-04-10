import os
import glob
import tempfile


"""
Script to process the time-variable ERA5 features.
The output is 36 files, one for each month of data.
See Workflow/ 1.1 ERA5 Data
"""




root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/'

#Get all file paths
ERA_skin_files = []
ERA_skt_files  = []
ERA_sfc_files  = []

for i in ['2018','2019','2020']: #Just get these three years
    skin_files_i = glob.glob(root+f'ERA_skin/*_{i}_*.grib')
    sfc_files_i  = glob.glob(root+f'ERA_sfc/*_{i}_*.grib')
    skt_files_i  = glob.glob(root+f'ERA_skt/*_{i}_*.grib')
    
    ERA_skin_files.extend(skin_files_i)
    ERA_sfc_files.extend(sfc_files_i)
    ERA_skt_files.extend(skt_files_i)
    
#Sort it
ERA_skin_files = sorted(ERA_skin_files)
ERA_skt_files = sorted(ERA_skt_files)
ERA_sfc_files = sorted(ERA_sfc_files)


#Define the features we are interested in from ERA_skin
time_variable_features = 'aluvp/aluvd/alnip/alnid/istl1/istl2/sd/2d/fal' #These are the features in ERA_skin that are not constant


#And IO paths
output_path_timevariable = root + 'processed_data/ERA_timevariable/'


for i in range(len(ERA_skin_files)): #Iterate over every monthly file
    
    
    ERA_sfc = ERA_sfc_files[i]
    ERA_skin = ERA_skin_files[i]
    ERA_skt = ERA_skt_files[i]
    
    outfile_timevariable = f'{output_path_timevariable}ERA_{i}.grib'
    print (i, outfile_timevariable)
    print (ERA_sfc)
    print (ERA_skin)
    print (ERA_skt)
    print('-----------------')
    with tempfile.NamedTemporaryFile() as tmp1, tempfile.NamedTemporaryFile() as tmp2: #Create two tmp files to write to
        
        print ('Created tmpfile')
        tmpfile1 = tmp1.name
        tmpfile2 = tmp2.name
        
        print('Extracting time variable features')
        #Extract the time variable features from ERA_skin, save to tmpfile
        query_extract = f'grib_copy -w shortName={time_variable_features} {ERA_skin} {tmpfile1}'
        os.system(query_extract)
    
        print('Setting the non surface levels to same type: istl1,2')
        # Deal with the istl1,2 features.  - not surface levels
        query_level = f'grib_set -s level=0 -w level=7 {tmpfile1} {tmpfile2}'
        os.system(query_level)
        
        print('Merging all time variable files')
        #Merge ERA_sfc, our processed ERA_skin tmpfile and ERA_skt into a single file
        query_merge = f'grib_copy {ERA_sfc} {tmpfile2} {ERA_skt} {outfile_timevariable}'
        os.system(query_merge)
        

    
