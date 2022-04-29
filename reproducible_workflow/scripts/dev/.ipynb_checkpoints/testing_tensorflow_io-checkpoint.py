import tensorflow as tf
import csv
#import tensorflow_io as tfio
#import os



def read_csv(filename):
    with open(filename, 'r') as f:
        next(f) #skip first header row
        for line in f.readlines():
            record = line.rstrip().split(',')
            features = [float(n) for n in record[6:10]]
            label = float(record[5])
            yield features, label


def get_dataset(filename):
    generator = lambda: read_csv(filename)
    n_features=4
    return tf.data.Dataset.from_generator(
        generator, (tf.float32, tf.int32), ((n_features,), ()))


filename = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/test_file.csv'


dataset = get_dataset(filename)
dataset = dataset.shuffle(100).batch(10000) #total nrows is 3,048,184

#maybe a pre-fetch here as well?


nfeatures = 4
#Create a basic NN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(int(nfeatures/2), activation='relu',input_shape=(nfeatures,),name='layer1'),
    tf.keras.layers.Dense(int(nfeatures/2), activation='relu',input_shape=(nfeatures,),name='layer2'),
    tf.keras.layers.Dense(1, name='output')
  ])

#Compile it
opt = tf.keras.optimizers.Adam()#(learning_rate=3e-4) 
model.compile(optimizer=opt,
              loss='mse',
              metrics=['accuracy'])




history = model.fit(dataset, 
                    epochs=2, 
                    verbose=1
                            ) 






print(dataset)


#Define input file
#root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/'
#input_file = f'{root}processed_data/joined_data/all_months_V3.h5'



### --- GPU configuration - dont use too much memory
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
#gpus = tf.config.list_physical_devices('GPU')
#print('GPUs:', gpus)
#if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
 #   try:
  #      tf.config.set_logical_device_configuration(
   #         gpus[0],
    #       [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
      #  logical_gpus = tf.config.list_logical_devices('GPU')
     #   print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #except RuntimeError as e:
     #   # Virtual devices must be set before GPUs have been initialized
      #  print(e)
### ---    





#print(tf.test.gpu_device_name())
#print(tf.config.list_physical_devices('GPU'))



#print ('---------d_train----------')
#d_train = tfio.IODataset.from_hdf5(input_file,dataset='df')



print ('All completed ok')




