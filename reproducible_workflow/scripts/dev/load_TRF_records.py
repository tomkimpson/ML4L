import tensorflow as tf
import glob


AUTOTUNE = tf.data.experimental.AUTOTUNE
fl = glob.glob('tfrecords/*')
shuffle_size=1024
print (fl)

files_ds = tf.data.Dataset.list_files(fl)
print('files_ds:', files_ds)

ds = tf.data.TFRecordDataset(files_ds,num_parallel_reads=AUTOTUNE)
print('ds:', ds)


ds = ds.shuffle(shuffle_size)
print(ds)
