

import tensorflow as tf
import numpy as np
import pandas as pd




input_filename = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/test_data.tfrecords'



#Get the raw data
filenames = [input_filename]
raw_dataset = tf.data.TFRecordDataset(filenames)


#print the first 10 items
for raw_record in raw_dataset.take(10):
    print(repr(raw_record))


print ('------------------------------------------------')
print ('------------------------------------------------')
print ('------------------------------------------------')


# Create a description of the features.
feature_description = {
    'feature': tf.io.FixedLenFeature([32, ], tf.float32,default_value=np.zeros(32,)),
    'label' : tf.io.FixedLenFeature([16, ], tf.float32,default_value=np.zeros(16,))
}

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)





parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=1)
print(parsed_dataset)

for parsed_record in parsed_dataset.take(10):
    print(repr(parsed_record))
    









# # Create a description of the features.
# feature_description = {
#     'feature': tf.io.FixedLenFeature(32, tf.float32, default_value=0),
#     'label': tf.io.FixedLenFeature(1, tf.float32, default_value=0)
# }

# def _parse_function(example_proto):
#   # Parse the input `tf.train.Example` proto using the dictionary above.
#   return tf.io.parse_single_example(example_proto, feature_description)


# parsed_dataset = raw_dataset.map(_parse_function)


# for parsed_record in parsed_dataset.take(10):
#     print(repr(parsed_record))
    
    
    
    
    
print ('-----------------------------------------------------')
print ('---------------------KERAS---------------------------')
print ('-----------------------------------------------------')

# shuffle_buffer = 20 #What to set this as?
# batch_size = 1024
# ignore_order = tf.data.Options()
# ignore_order.experimental_deterministic = False  # disable order, increase speed


# dataset = tf.data.TFRecordDataset(filenames)
# dataset = dataset.with_options(ignore_order)  # uses data as soon as it streams in, rather than in its original order
# dataset = dataset.shuffle(shuffle_buffer)#.repeat()  # shuffle and repeat
# dataset = dataset.map(_parse_function, num_parallel_calls=4)
# dataset = dataset.batch(batch_size).prefetch(2)  # batch and prefetch

# # #Create a basic NN model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(4, activation='relu',input_shape=(4,),name='SLAYER1'),
#     tf.keras.layers.Dense(1, name='output')
# ])

# # #Compile it
# opt = tf.keras.optimizers.Adam(learning_rate=3e-4) 
# model.compile(optimizer=opt,
#             loss='mse',
#             metrics=['accuracy'])

# history = model.fit(dataset, 
#                     epochs=100, batch_size=1024,
#                     verbose=1
#                     ) 