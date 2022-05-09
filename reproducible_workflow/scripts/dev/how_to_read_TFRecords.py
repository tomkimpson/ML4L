

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
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)


parsed_dataset = raw_dataset.map(_parse_function)


for parsed_record in parsed_dataset.take(10):
    print(repr(parsed_record))
    
    
    
    
    
print ('-----------------------------------------------------')
print ('---------------------KERAS---------------------------')
print ('-----------------------------------------------------')

shuffle_buffer = 20 #What to set this as?
batch_size = 1024
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False  # disable order, increase speed


dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.with_options(ignore_order)  # uses data as soon as it streams in, rather than in its original order
dataset = dataset.shuffle(shuffle_buffer).repeat()  # shuffle and repeat
dataset = dataset.map(_parse_function, num_parallel_calls=4)
dataset = dataset.batch(batch_size).prefetch(2)  # batch and prefetch

print (dataset)