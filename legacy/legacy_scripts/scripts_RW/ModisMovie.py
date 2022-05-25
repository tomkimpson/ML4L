import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib.colors as mc
import matplotlib.colorbar as cb
import pandas as pd



path = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/processed_data/trained_models/ML_acc386a38f1d481da481cf2bffb5bfba/' #Path to ML folder of predictions



list_of_files = glob.glob(path+'')


# init the figure
fig,ax1= plt.subplots(nrows=1,figsize=(30, 20))
fig.subplots_adjust(right=0.8,wspace=0, hspace=0)
cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
cmap = plt.cm.viridis


norm = mc.Normalize(vmin=200, vmax=300)

cb1 = cb.ColorbarBase(cbar_ax, cmap=cmap,
                                 norm=norm,
                                 orientation='vertical')


def update(i):
    # clear the axis each frame
    ax1.clear()
    ax2.clear()
    
 
    
    #Get the data
    ti=t[i]
    condition = f"time_daily == '{ti}'"
    ds = ds_map.query(condition)
    
    x = ds.longitude_ERA
    y = ds.latitude_ERA
    
    z1 = ds['error_ERA_abs']
    z2 = ds['error_NN_abs']
        
    
    # replot things
    sc = ax1.scatter(x, y,
               s=1,
               c=cmap(norm(z1)),
               linewidths=1, alpha=.7
               )
    
    sc = ax2.scatter(x, y,
               s=1,
               c=cmap(norm(z2)),
               linewidths=1, alpha=.7
               )
 

    ax1.set_title(str(ti) +' error_ERA' )
    ax2.set_title(str(ti) +' error_NN' )



print('Animating')
ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)
print('Saving')
ani.save('test.mp4')