#Pipeline script to link processed ERA/MODIS data as described in notebook 3.


import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np



def nearest_neighbour(X,reference):
    #For each entry in X, find the closest match in reference
    
    NN = NearestNeighbors(n_neighbors=1, metric='haversine') #algorithm = balltree, kdtree or brute force. Automatically selected based on data
    
    NN.fit(np.deg2rad(reference[['y', 'x']].values)) 

    query_lats = X['y'].astype(np.float64)
    query_lons = X['x'].astype(np.float64)
    Y = np.deg2rad(np.c_[query_lats, query_lons])

    distances, indices = NN.kneighbors(Y, return_distance=True)
    
    r_km = 6371 # multiplier to convert to km (from unit distance)
    distances = distances*r_km
    
    return distances,indices


def join_and_filter(df,df_ref,idx,distances,tolerance):
    #Join a df with a reference df, defined via indices idx, and then filter by distance c.g. tolerance
    

    #Combine both dfs
    df_joined = df.reset_index().join(df_ref.iloc[idx.flatten()].reset_index(), lsuffix='_ERA',rsuffix='_MODIS')
    df_joined['distance'] = distances
    df_joined['MODIS_idx'] = idx

    #Filter and surface selected columns
    return df_joined.query('distance < %.9f' % tolerance)


def pipeline(df_ERA,df_MODIS):
    
    #Crop ERA df by location
    string_query = 'x > %.9f & x < %.9f' % (min(df_MODIS['x']),max(df_MODIS['x']))
    df_ERA_selected = df_ERA.query(string_query) #Just get ERA points within MODIS longitude bounds
    
    # Setup Nearest Neighbours using MODIS as reference dataset
    # Use Haversine calculate distance between points on the earth from lat/long
    distances, indices = nearest_neighbour(df_ERA_selected,df_MODIS)
    

    #Join based on these indices to create a single df
    df_out = join_and_filter(df_ERA_selected,df_MODIS,indices,distances,10) #Get haversines < 10 km



    return df_out



#Load the each of the dataframes
print ('Loading data')
df_ERA_processed = pd.read_pickle('../jupyter_notebooks/data/ERA_processed.pkl')
df_MODIS_processed = pd.read_pickle('../jupyter_notebooks/data/MODIS_processed.pkl')



#Select a day
df_ERA_day = df_ERA_processed.query('time < "2018-01-02 00:00:00"')

#Reindex for convenience
df_ERA_day = df_ERA_day.reset_index()
df_MODIS_processed = df_MODIS_processed.reset_index()

#Get a list of unique times
unique_times = df_ERA_day.time.unique()

#Separate 24hr df into collection of dfs, each for 1 hour
dfs24_ERA   = [df_ERA_day[df_ERA_day['time'] == s] for s in unique_times]
dfs24_MODIS = [df_MODIS_processed[df_MODIS_processed['time'] == s] for s in unique_times]


import time

t0 = time.time()



#Process each hour
print ('Begin process')
dfs24 = []
for i in range(len(dfs24_ERA)):
    print(i)
    ERA1 = dfs24_ERA[i]
    MODIS1 = dfs24_MODIS[i]
    df_out = pipeline(ERA1,MODIS1)
    dfs24.append(df_out)

t1 = time.time()
print ('Processing time for 24 hours was:', t1-t0, ' seconds')
    
#Save each hour
counter = 0
for i in dfs24:
    fname = 'data/'+str(counter)+'.pkl'
    i.to_pickle(fname)
    counter += 1

