

import tensorflow as tf
import numpy as np
import pandas as pd
import glob

root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/'


SHUFFLE_SIZE = 1024
BATCH_SIZE = 1024
AUTOTUNE = tf.data.AUTOTUNE
num_features = 40
training_filenames = glob.glob(f'{root}training_data_with_monthly_lakes/TFRecords/*.tfrecords')
validation_filenames = glob.glob(f'{root}validation_data_with_monthly_lakes/TFRecords/*.tfrecords')

def _parse_function(example_proto):
    
    feature_description = {
    'feature_input': tf.io.FixedLenFeature([num_features,], tf.float32,default_value=np.zeros(num_features,)),
    'label' : tf.io.FixedLenFeature([1, ], tf.float32,default_value=np.zeros(1,))
    }
  
    example = tf.io.parse_single_example(example_proto, feature_description)

    image = example['feature_input']
    label = example['label']
    

    return image,label



def get_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames).map(_parse_function)  # automatically interleaves reads from multiple files
    dataset = dataset.shuffle(SHUFFLE_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


train_dataset = get_dataset(training_filenames)
valid_dataset = get_dataset(validation_filenames)

print(f'{root}training_data_with_monthly_lakes/TFRecords/*.tfrecord')
print(training_filenames)
print(validation_filenames)



# print(train_dataset)


#Create a basic NN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='relu',input_shape=(num_features,),name='layer1'),
    tf.keras.layers.Dense(1, name='output')
])

# #Compile it
opt = tf.keras.optimizers.Adam(learning_rate=3e-1) 
model.compile(optimizer=opt,
            loss='mse',
            metrics=['accuracy'],
             run_eagerly=True)



history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=2
)










print ('Completed OK')

































# #print the first 10 items
# for raw_record in raw_dataset.take(10):
#     print(repr(raw_record))


# print ('------------------------------------------------')
# print ('------------------------------------------------')
# print ('------------------------------------------------')


# # Create a description of the features.



# for parsed_record in parsed_dataset.take(10):
