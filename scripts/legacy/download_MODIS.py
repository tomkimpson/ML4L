from config import *
import os
import sys
#Create a text file that can be read by wget
def create_text_file(years,months):


    base = 'https://e4ftl01.cr.usgs.gov/MOLT/MOD11C3.006/' 
    
    f = open(wget_file,'w')
    for y in years:
        for m in months:
            date = y+'.'+m+'.01/ \n'
            string = base+date
            f.write(string)
    f.close()


if load_y_data_from_remote:

    create_text_file(years,months)
    #Not the most Pythonic way to do this, but does the job for now
#    os.system('wget -r -l1 --no-parent -A "*.hdf" -i '+wget_file + ' -P " '+data_root+'" -q --show-progress')

    msg = 'wget -r -l1 --no-parent -A "*.hdf" -i '+wget_file+' -P'+data_root+' -q --show-progress'
    print (msg)
    os.system(msg)
