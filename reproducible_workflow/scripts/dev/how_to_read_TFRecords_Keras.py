

import tensorflow as tf
import numpy as np
import pandas as pd

# n_observations = int(1e4)

# feature0 = np.random.randn(n_observations)
# feature1 = np.random.randn(n_observations)
# feature2 = np.random.randn(n_observations)
# ds = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2))

from tensorflow.keras.layers import Dense, DenseFeatures
from tensorflow.feature_column import numeric_column


# inputs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# outputs= [ 0.0, 0.1, -0.1]

# ds = tf.data.Dataset.from_tensors((
#     {'dense_input':inputs },outputs ))
# print(ds)
# ds = ds.repeat(32).batch(32)
# print(ds)
# model = tf.keras.models.Sequential(
#     [
#       tf.keras.Input(shape=(7,), name='dense_input'),
#       tf.keras.layers.Dense(20, activation = 'relu'),
#       tf.keras.layers.Dense(3, activation = 'linear'),
#     ]
# )

# model.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])

# model.fit(ds, epochs = 1)


# <BatchDataset shapes: {feature_input: (None, 51), label: (None, 1)}, types: {feature_input: tf.float32, label: tf.float32}>

# <BatchDataset shapes: ({dense_input: (None, 7)}, (None, 3)), types: ({dense_input: tf.float32}, tf.float32)>






SHUFFLE_SIZE = 1024
BATCH_SIZE = 1024
AUTOTUNE = tf.data.AUTOTUNE



def _parse_function(example_proto):
    
    feature_description = {
    'feature_input': tf.io.FixedLenFeature([51,], tf.float32,default_value=np.zeros(51,)),
    'label' : tf.io.FixedLenFeature([1, ], tf.float32,default_value=np.zeros(1,))
    }
  
    example = tf.io.parse_single_example(example_proto, feature_description)

    image = example['feature_input']
    label = example['label']
    
    print(image.shape)
    print(label.shape)
    
    #image = tf.cast(image, tf.float32)
    #print('IMAGEs')
    #print(image)
    #image = tf.reshape(image, [None,51 ])
    

    
    return image,label



TRAINING_FILENAMES = ['/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/ECMWF_files/raw/processed_data/joined_data/test_data.tfrecords']
print(TRAINING_FILENAMES)
ds = tf.data.TFRecordDataset(TRAINING_FILENAMES).map(_parse_function)  # automatically interleaves reads from multiple files



ds = ds.shuffle(SHUFFLE_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds = ds.batch(BATCH_SIZE)

print(ds)




# def load_dataset(filenames):
#     ignore_order = tf.data.Options()
#     ignore_order.experimental_deterministic = False  # disable order, increase speed
#     dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files
#     dataset = dataset.with_options(ignore_order)  # uses data as soon as it streams in, rather than in its original order
#     dataset = dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
#     return dataset



# def get_dataset(filenames):
#     dataset = load_dataset(filenames)
#     dataset = dataset.shuffle(SHUFFLE_SIZE)
#     dataset = dataset.prefetch(buffer_size=AUTOTUNE)
#     dataset = dataset.batch(BATCH_SIZE)
#     return dataset



# train_dataset = get_dataset(TRAINING_FILENAMES)



# print(train_dataset)


#Create a basic NN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='relu',input_shape=(51,),name='layer1'),
    tf.keras.layers.Dense(1, name='output')
])

# #Compile it
opt = tf.keras.optimizers.Adam(learning_rate=3e-4) 
model.compile(optimizer=opt,
            loss='mse',
            metrics=['accuracy'])



history = model.fit(
    ds,
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
